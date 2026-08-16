"""Headless console engine for GUI3.

Everything the desk can do goes through ``ConsoleEngine.dispatch``, so the
web front end, the tests, and scripts drive identical code. Only the public
``hifi_anova`` API is used for fitting; a few analysis helpers are imported
from ``hifi_anova.analysis``/``hifi_anova.core`` — flagged for promotion to
the public API (design doc §7, architecture doc "flagged private imports"):

- ``analysis.component_eval.frequency_decomposition`` (DEGREES view)
- ``analysis.interaction_discovery.scan_missing_pairs`` (routing scan)
- ``analysis.automl.ridge_analytics`` + ``result._fitted_design`` (PathService:
  the λ₁ criterion path is recomputed from the design the trainer solved)
- ``core.features.basis_size`` (compile shape keys)
- RESIDUAL bus (C16): ``training.analytic_residual.create_residual`` /
  ``_create_fitted_residual`` (underscore-private), ``core.projection.
  project_features_orthogonal``, ``training.hyperopt.RidgePathEigSolver``
  incl. its ``_mu``/``_Q``/``_Phi`` internals (per-λ residual-block leverages
  in O(N·M)) — all BR-08 promotion candidates

Concurrency: ONE worker thread consuming a job queue (JAX precision is
process-global → single worker is a rule). Jobs are real fits or low-priority
"warm" fits that only pre-compile a neighboring model shape. Cooperative
cancel via ``should_stop``; a cancelled or failed fit never replaces the
current view; warm fits never touch the view at all.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .datasets import REGISTRY, load as load_dataset, make_fatigue_demo  # noqa: F401
from .planner import plan_first_try, default_k2_cap

DEFAULT_K = 3          # desk default (schema default K1=5 is too slow for live faders)
# λ defaults mirror the backend config schema (lambda_order1 / lambda_h /
# lambda_order2) — pinned by test_defaults_match_backend_schema, so they cannot
# drift from the library without a test failing (no JAX import at module load).
LAMBDA1_DEFAULT = 1e-3
LAMBDA_H_DEFAULT = 0.1
LAMBDA2_DEFAULT = 1e-2  # pair (order-2) penalty amount; backend default lambda_order2
# Control limits — the single source of truth for engine AND client: shipped to
# the browser in snapshot()['limits'], so the fader/slider ranges cannot drift
# from what the engine accepts (they used to: fader 1–8 vs engine 1–12).
K_MIN, K_MAX = 1, 12   # per-channel fidelity fader (schema: K1 >= 1, no library max)
LAMBDA_MIN, LAMBDA_MAX = 1e-9, 1e4  # hard accept range of cmd_set_lambda
PAIR_K2_CAP = 5        # cap pair fidelity K2 (columns grow as K2²) for interactivity
VAR_K2H_DEFAULT = 2    # BR-05 default 2nd-order VARIANCE order when a var pair is on
# EFFECTS "2nd ORDER = ALL" guardrail: entering the full pair clique is refused
# when the projected order-2 column count C(a,2)·B₂² would exceed this
# (interactivity + conditioning; the message suggests SELECT / lower PAIRS K).
ORDER2_ALL_MAX_COLS = 3000
# Default-fit d-rule (auto_order2, opt-in like warm_neighbors — the server
# enables it, tests don't): on a NEW dataset the quick fit includes ALL pairs
# up to this many variables, stages a budgeted TOP clique (after the first
# first-order fit provides the Sᶠ ranking) up to the second bound, and stays
# honestly first-order above it. Compute-time defaults, not statistical claims.
ORDER2_AUTO_ALL_MAX_D = 4
ORDER2_AUTO_TOP_MAX_D = 10
# INITIAL TRY (first-shot budget-ladder planner, gui3/planner.py): a modular,
# opt-in heuristic (server default: balanced; tests/scripts: off) that plans
# the FIRST fit's k1 / pair count / K2 from (N, D, Z) — Z is the speed preset
# in total design columns. Coexists with the d-rule: when the planner is ON it
# owns the install-time default; OFF restores the d-rule/TOP behavior exactly.
# Pragmatic compute-time defaults, disclosed — NOT reviewed statistics (A.6
# brief material, same class as the TOP budget above).
PLAN_SPEEDS = ("off", "fast", "balanced", "thorough")
PLAN_Z = {"fast": 60, "balanced": 150, "thorough": 400}
PLAN_GAMMA = 3       # min rows per design column (ridge-safe)
PLAN_MIN_PAIRS = 3   # fewer affordable pairs than this -> don't bother
PAIR_THUMB_G = 12      # resolution of the per-pair interaction-component thumbnail
PAIR_THUMB_MAX = 12    # cap thumbnails per fit (each is a predict-on-grid)
# λ₁ path grid (log10 endpoints = the console's λ slider display range)
PATH_LOG10_MIN, PATH_LOG10_MAX, PATH_POINTS = -6.0, 2.0, 45
WARM_MAX_JOBS = 6
# ΔLOO TEST (R25, rank-only half) fan-out bounds — the cap is LOGGED in the
# result (never a silent truncation, S2/S3 rule)
LOO_TEST_MAX_DROPS = 16
LOO_TEST_MAX_ADDS = 12
CSV_MAX_N = 4000  # row cap for browser CSV import (interactivity)
# RESIDUAL bus (C16, X12C Phase 1): a post-hoc smooth catch-all fitted on the
# CURRENT result's leftover — the backend's linear Stage-C families, core-legal
# per DEC-057. The orthogonal projection guarantees the main fit's coefficients
# (and Sobol shares) are untouched, so this NEVER reruns Stages A/B.
RESIDUAL_FAMILIES = ("nystrom", "rbf", "rff")
RESIDUAL_M_MAX = 1000            # feature-count cap (the path pays one M×M eigh)
RES_LAMBDA_MIN, RES_LAMBDA_MAX = 1e-9, 1e9  # hard accept range of lam=
# λ_res criterion-path grid (log10 endpoints = the RESIDUAL fader's range)
RES_PATH_LOG10_MIN, RES_PATH_LOG10_MAX, RES_PATH_POINTS = -6.0, 6.0, 49
# family -> (M config key, width config key, library default width). The desk's
# M/WIDTH faders map onto each family's own hyperparameters; defaults are the
# library's shipped values (fixed compute defaults, NOT reviewed statistics).
RESIDUAL_PARAM_KEYS = {
    "rbf": ("n_centers", "sigma", 0.2),
    "rff": ("n_features", "gamma", 3.0),
    "nystrom": ("n_inducing", "lengthscale", 0.2),
}
# family -> library-default feature count (M FEATURES fader AUTO), capped at
# the train-row count in the library itself.
RESIDUAL_M_DEFAULT = {"rbf": 300, "nystrom": 300, "rff": 1000}
# WIDTH profile grid (Phase 1.5, analysis §4): log-spaced multipliers of the
# library-default width. A disclosed COMPUTE default — the grid choice is
# §6.4 brief material, never blessed statistics. Each grid point pays one
# Z rebuild + projection + M×M eigh + the λ path, so the profile is capped by
# M (RFF's default M=1000 would pay ~9 seconds of eighs) and cached per fit.
RES_WIDTH_GRID_FACTORS = tuple(float(v) for v in np.logspace(-0.7, 0.7, 9))
RES_WIDTH_PROFILE_M_MAX = 500
# WIDTH fader/overlay display range (log10 of the width value) — shipped in
# snapshot limits so the client fader and the profile overlay cannot drift
# from the engine's grid (the Ses03 audit's documented gap, closed Ses04…
# well, this pass). Engine still ACCEPTS any width > 0 via the API.
RES_WIDTH_LOG10_MIN, RES_WIDTH_LOG10_MAX = -1.5, 1.5
# NYSTRÖM kernel selector (Matérn ν stepper, analysis §4) — the backend's own
# kernel_type surface (model/linear_residual.NystromResidual). A user CHOICE
# like the family select (library default rbf), not a tuned proposal.
RES_KERNELS = ("rbf", "matern32", "matern52")
# Residual/combined total-order Sobol readout (Phase 1.5, analysis §5c):
# Jansen/Saltelli pick-freeze on a scrambled-Sobol QMC design under the
# UNIFORM input measure on the model cube — n(D+2) model evaluations, sized
# down with D to stay interactive (disclosed in the block). Model-based,
# EXPLORATORY, never an admission criterion; normalization defaults are §6.8.
RES_SOBOL_N_BY_D = ((8, 4096), (16, 2048), (10 ** 9, 1024))
# Interpretive thresholds — single source of truth for engine AND client
# (shipped in snapshot()['thresholds']; the report uses the same values). These
# used to be scattered as literals in the page script, i.e. unaudited
# statistical claims living in presentation code.
CORR_WARN = 0.3        # A1 lamp: |input corr| at/above -> shares conditional
GAP_WARN = 0.05        # R18 two-fit gap lamp: |efficient-interpretable| warn
ROLE_SF_MIN = 0.02     # role tag: S_f above -> channel counts as MEAN
ROLE_SH_MIN = 0.05     # role tag: S_h above -> channel counts as VAR
KTRIM_TAIL = 0.03      # AUTO ghost: top-degree energy below this -> propose K-1
PRED_SH_WEIGHT = 0.4   # provisional PRED* weight on S_h (pending BlockDelta)
NOISE_AGREE_SPREAD = 1.25  # noise-leg triangulation: max/min above -> DISAGREE
# Data-quality flag — mirrors the backend's discrete/heavily-tied-column
# criterion (hifi_anova.data.preprocessing._TIES_MIN_UNIQUE/_TIES_MAX_SHARE,
# pinned by test; literals here to keep module import JAX-free) so the desk
# warns at IMPORT time with the same rule the fit will warn with.
TIES_MIN_UNIQUE = 10
TIES_MAX_SHARE = 0.2


def _model_free_sigma2(X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
    """Model-free noise variance — the third, fit-independent noise leg.

    Two estimators, both free of any fitted mean model, so they triangulate
    against the RSS/df and σ²(λ)-min legs (which both come from the fit):

    - **replicate**: if the design has exact duplicate X rows, the pooled
      within-replicate variance of y is an unbiased σ² with no smoothness
      assumption at all. Used when enough replicate groups exist.
    - **Rice / nearest-neighbour**: otherwise, ½·mean (y_i − y_{nn(i)})² over
      each point's nearest neighbour in standardized X (the classic
      difference-based estimator). Assumes only local continuity of the mean.

    Returns ``{sigma2, share, method, n}`` (share = σ²/Var(y)) or ``None`` when
    there are too few rows to estimate.
    """
    y = np.asarray(y, float).reshape(-1)
    X = np.asarray(X, float)
    n = len(y)
    if n < 10:
        return None
    var_y = float(np.var(y))
    # --- replicate leg: exact duplicate rows -> pooled within-group variance
    try:
        _, inv, counts = np.unique(np.round(X, 12), axis=0,
                                   return_inverse=True, return_counts=True)
        rep_groups = int(np.sum(counts >= 2))
        rep_rows = int(np.sum(counts[counts >= 2]))
        if rep_groups >= 5 and rep_rows >= 0.05 * n:
            ss = 0.0
            df = 0
            for g in np.nonzero(counts >= 2)[0]:
                yg = y[inv == g]
                ss += float(np.sum((yg - yg.mean()) ** 2))
                df += len(yg) - 1
            if df > 0:
                s2 = ss / df
                return {"sigma2": s2,
                        "share": (s2 / var_y) if var_y > 0 else None,
                        "method": "replicate", "n": rep_rows}
    except Exception:
        pass
    # --- Rice / nearest-neighbour difference estimator
    try:
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - X.mean(axis=0)) / sd
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(Xs)
        _, idx = nn.kneighbors(Xs)
        diffs = y - y[idx[:, 1]]
        s2 = 0.5 * float(np.mean(diffs ** 2))
        return {"sigma2": s2, "share": (s2 / var_y) if var_y > 0 else None,
                "method": "nn", "n": n}
    except Exception:
        return None


def _data_quality(X: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    """Discrete / heavily-tied input columns, flagged at dataset install.

    Same criterion the backend applies to the train split in
    ``preprocess_data``: fewer than ``TIES_MIN_UNIQUE`` distinct values, or one
    value in more than ``TIES_MAX_SHARE`` of rows. The quantile transform sends
    tied values to a single atom, so the analytic Sobol shares of such a
    channel are unreliable — the desk should say so BEFORE the fit does."""
    out: List[Dict[str, Any]] = []
    X = np.asarray(X, float)
    n = X.shape[0]
    if n == 0:
        return out
    for j in range(X.shape[1]):
        vals, counts = np.unique(X[:, j], return_counts=True)
        if len(vals) < TIES_MIN_UNIQUE or counts.max() > TIES_MAX_SHARE * n:
            out.append({"name": names[j] if j < len(names) else str(j),
                        "n_unique": int(len(vals)),
                        "max_share": _f(counts.max() / n)})
    return out


def _max_abs_corr(X: np.ndarray, names: List[str]) -> Optional[Dict[str, Any]]:
    """A1 honesty leg: the largest |Pearson correlation| between any two input
    columns. Sobol attribution assumes (near-)independent inputs; a high value
    means shares are conditional on the correlation and should be read with
    caution. Returns ``{value, pair:[a,b]}`` or ``None`` (D<2 / degenerate)."""
    X = np.asarray(X, float)
    if X.shape[1] < 2 or X.shape[0] < 3:
        return None
    try:
        C = np.corrcoef(X, rowvar=False)
        C = np.abs(np.asarray(C, float))
        np.fill_diagonal(C, 0.0)
        if not np.isfinite(C).any():
            return None
        C[~np.isfinite(C)] = 0.0
        i, j = np.unravel_index(int(np.argmax(C)), C.shape)
        return {"value": float(C[i, j]),
                "pair": [names[i], names[j]] if names else [int(i), int(j)]}
    except Exception:
        return None


def _sobol_gap_max(res) -> Optional[Dict[str, Any]]:
    """R18 two-fit gap honesty leg. ``res.sobol_gap`` (populated ONLY on a Stage-D
    heteroscedastic fit) reports, per first-order variable, the difference between
    the *efficient* (precision-weighted / GLS predictive) Sobol share and the
    *interpretable* (unit-weight) share the rack meters show (two-fit convention,
    DEC-030). A large gap flags that the predictive fit attributes variance
    differently from the reported interpretable attribution. Returns the strongest
    such gap ``{value, name}`` (signed value, efficient−interpretable) or ``None``
    on a homoscedastic fit / when no gap surface exists."""
    gap = getattr(res, "sobol_gap", None)
    if not gap:
        return None
    first = gap.get("first_order", {}) if isinstance(gap, dict) else {}
    if not first:
        return None
    try:
        name, g = max(first.items(), key=lambda kv: abs(float(kv[1])))
    except (ValueError, TypeError):
        return None
    return {"value": _f(g), "name": str(name)}


def _f(x) -> Optional[float]:
    """JSON-safe float (None passes through)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


class ConsoleEngine:
    def __init__(self) -> None:
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.names: List[str] = []
        self.dataset_label = ""
        self.dataset_id = ""
        # mix settings
        self.k: Dict[str, int] = {}
        self.muted: set = set()
        self.lambda1 = LAMBDA1_DEFAULT
        self.lambda_h = LAMBDA_H_DEFAULT
        self.lambda2 = LAMBDA2_DEFAULT
        # admitted interactions (ROUTE / M3): a set of frozenset({a, b}) variable
        # NAME pairs the desk has "patched in". The backend admits pairs only as an
        # active-variable clique ('both' mode) — a single pair is exact, disjoint
        # pairs induce the clique closure (surfaced, never silently reshaped).
        self.pairs: set = set()
        # interaction fidelity K2 (harmonic order of EVERY pair block). The backend
        # takes a single scalar K2 for all pairs — there is no per-pair K2 — so this
        # is one global control. None = auto (min(maxK, PAIR_K2_CAP)); a slider sets
        # an explicit value, capped at PAIR_K2_CAP (columns grow as K2²).
        self.pair_k2: Optional[int] = None
        # EFFECTS order-2 mode: 'off' (first-order only), 'all' (full pair
        # clique over the active channels, auto-tracked on mute/unmute), or
        # 'select' (hand-patched pairs via ROUTE). 'all' is just the clique the
        # backend natively admits — pairs stays the single source of truth.
        self.order2 = "off"
        # TOP mode detail: which channels the budgeted clique spans (heuristic
        # preselection by last-fit first-order Sᶠ rank — disclosed, exploratory)
        self._order2_top: Optional[Dict[str, Any]] = None
        self.basis = "legendre"
        self.strategy = "auto"  # penalty shape (TONE); auto = backend default
        self.hetero = False
        self.auto_reset = True  # a NEW dataset resets tuning knobs to defaults
        # fit state
        self.status = "EMPTY"  # EMPTY / READY / FITTING / FIT_READY / ERROR
        self.stale = False
        self.error: Optional[str] = None
        self.result = None
        self.view: Dict[str, Any] = {}
        self.scan: Optional[Dict[str, Any]] = None
        self.gate: Optional[Dict[str, Any]] = None  # group-lasso channel selection
        # VERIFY LOO (Ses07): Tier-III oracle result for the CURRENT fit —
        # cleared on any stale-marking change (it verifies one fitted model)
        self.verify: Optional[Dict[str, Any]] = None
        self._verify_pending = False
        # ΔLOO TEST (R25, rank-only): ranked paired-ΔLOO of candidate structure
        # changes vs the CURRENT fit — cleared on any stale change (each Δ is a
        # difference against one fitted model). Rank only: NO keep/drop rule
        # (the stopping rule is expert-gated, X9C_gui3_loo_selection_brief.md).
        self.loo_test: Optional[Dict[str, Any]] = None
        self._loo_test_pending = False
        # RESIDUAL bus (C16): the post-hoc smooth catch-all fitted on the
        # CURRENT result — cleared on any stale change (it belongs to the fit
        # it was computed on). ``residual_cfg`` is the CONTROL state (family +
        # fader values; None = library default / auto-λ) — it survives refits
        # so TAKES/PROFILE can carry it; the fitted block does not.
        self.residual: Optional[Dict[str, Any]] = None
        self._residual_pending = False
        self.residual_cfg: Dict[str, Any] = {
            "family": "nystrom", "n_centers": None, "width": None, "lam": None,
            "kernel": None}
        # the model WITHOUT the residual (for detach); None = nothing attached
        self._residual_base_model = None
        # WIDTH-profile cache: {"key": (epoch, family, n_centers), "block": …}
        # — the profile depends on the fit + family + M only (it spans all λ
        # and all widths), so λ/width fader refits reuse it. Cleared on stale.
        self._res_width_profile: Optional[Dict[str, Any]] = None
        self.noise_model_free: Optional[Dict[str, Any]] = None  # leg 3 (dataset)
        self.max_corr: Optional[Dict[str, Any]] = None          # A1 honesty (R17)
        self.data_quality: List[Dict[str, Any]] = []  # tied/discrete columns
        self.loo_samples: List[Dict[str, float]] = []
        self.last_fit_seconds: Optional[float] = None
        self.events: "queue.Queue[dict]" = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._yhat: Optional[np.ndarray] = None
        self._xt: Optional[np.ndarray] = None  # active X in the model's [0,1] space
        self._active_names: List[str] = []
        # job worker (fits + shape warmups share ONE thread — JAX rule)
        self._jobs: "queue.Queue[tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._active_job: Optional[str] = None  # 'fit' | 'warm' | None
        self._fit_queued = False
        # job-TERMINAL events queued by workers via _emit_done and flushed by
        # _job_loop only AFTER the job state settles (worker-thread-only list)
        self._post_events: List[Dict[str, Any]] = []
        # latency machinery
        self._seen_shapes: set = set()   # compile-shape keys already fitted
        self._epoch = 0                  # bumped on dataset install
        self._last_touched: Optional[str] = None  # last K-fader channel
        self.warm_neighbors = False      # server enables; tests opt in
        self.warming: Optional[str] = None  # label of the shape being warmed
        self._pairs_forced_uniform_k = False  # pairs + mixed K → uniform (surfaced)
        # default-fit d-rule (see ORDER2_AUTO_*): server enables; tests opt in
        self.auto_order2 = False
        self._auto_order2_pending = False  # 5..10 vars: TOP after first ranking
        # Array backend for fits AND desk analytics (numpy exact core):
        # 'jax' (default — byte-identical to before), 'numpy', or 'auto'
        # (numpy for everything the desk does — its configs never touch the
        # JAX-native residual/NN paths). The server sets 'auto'; tests keep
        # 'jax' so engine behavior/timing baselines are unchanged.
        self.fit_backend = "jax"
        # INITIAL TRY first-shot planner (see PLAN_*): off in tests/scripts,
        # the server sets a speed preset; "off" restores d-rule/TOP behavior
        self.plan_speed = "off"
        self._plan: Optional[Dict[str, Any]] = None  # last applied plan detail
        self._auto_plan_pending = False  # screened pairs staged after ranking
        # the ACTUAL kwargs / active columns of the last successful fit — so PRINT's
        # repro script reflects what really ran (incl. a mixed-K→uniform backend
        # fallback), not a fresh re-derivation that might differ.
        self._last_fit_kwargs: Optional[Dict[str, Any]] = None
        self._last_active: Optional[List[int]] = None
        # BR-06 (DEC-054) order-selective membership: name -> orders ⊆ {1, 2};
        # absent = both (the default). ``[2]`` = pair-only (NON-HIERARCHICAL: the
        # first-order block is excluded, its df not spent, the pair share absorbs
        # any true marginal); ``[1]`` = marginal-only (drops every pair touching
        # the variable). Only bites a second-order fit — surfaced as a label.
        self.term_orders: Dict[str, List[int]] = {}
        # BR-13: stash of the membership map as it was when the 1st-INDIVIDUAL
        # rocker staged the INTERCEPT-ONLY base — restored when it comes back
        # on (None = nothing staged via the rocker)
        self._prev_term_orders: Optional[Dict[str, List[int]]] = None
        # BR-01 (DEC-054) independent variance-side mute: names asserted
        # VARIANCE-FLAT (dropped from the Stage-D first-order variance model,
        # Sʰ ≡ 0 by assertion, df not spent) while KEPT in the mean model. Maps
        # to ``variance_variables`` = the active mean vars minus this set. Only
        # bites a heteroscedastic fit; an assertion of homoscedasticity along
        # the variable, so surfaced as a label.
        self.var_muted: set = set()
        # BR-04 (DEC-053) per-pair interaction fidelity: frozenset(name,name) ->
        # K2 override. When ANY override is set the pair fit switches to the
        # K2-mapping form ({(i,j):K2}), which PINS the exact admitted pairs (no
        # clique-induced extras) at their own harmonic orders. Empty = the
        # single global ``self.pair_k2`` scalar (unchanged default).
        self.pair_k2_map: Dict[frozenset, int] = {}
        # BR-05 (DEC-053) second-order VARIANCE: admitted interactions that carry
        # a variance (Sʰ) term, as frozenset(name,name) in ``var_pairs``, at the
        # scalar variance-pair order ``var_k2h``. Threads to ``K2h`` +
        # ``var_pair_selection``; only bites a heteroscedastic fit and only for
        # pairs whose endpoints are variance-active (not var_muted).
        self.var_pairs: set = set()
        self.var_k2h: int = VAR_K2H_DEFAULT
        # K₂-fader-at-zero pair mute: requested pairs whose interaction term is
        # muted (dropped from the fit; the pair stays admitted in ROUTE and its
        # strip stays visible at 0). A live mute forces the K2-mapping pinning
        # form — pair_selection alone could not exclude it (the backend's clique
        # closure would re-admit the pair).
        self.pair_muted: set = set()

    # ------------------------------------------------------------- commands
    def _abx(self):
        """The array-backend scope for desk work (fits, scans, surfaces,
        probes): 'numpy'/'auto' route the flagged backend-module calls the
        engine makes OUTSIDE the one-call API (component_eval, scan,
        frequency_decomposition, …) through the exact core too — otherwise
        they would fall back to eager JAX and re-pay per-shape compiles."""
        from hifi_anova.array_backend import use_array_backend
        return use_array_backend(
            "numpy" if self.fit_backend in ("numpy", "auto") else "jax")

    def dispatch(self, cmd: str, **args) -> Dict[str, Any]:
        try:
            handler = getattr(self, f"cmd_{cmd}", None)
            if handler is None:
                return {"ok": False, "error": f"unknown command: {cmd}"}
            with self._abx():
                return handler(**args)
        except Exception as exc:  # command errors must never kill the server
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def cmd_load_demo(self, n: int = 4000, seed: int = 42) -> Dict[str, Any]:
        return self.cmd_load_dataset("fatigue", n=n, seed=seed)

    def cmd_load_dataset(self, name: str, n: Optional[int] = None,
                         seed: int = 42) -> Dict[str, Any]:
        # a different registry id (or first load) = a genuinely new dataset;
        # re-loading the SAME id with a different N cap is NOT (preserve the mix)
        is_new = (self.dataset_id != name) or (self.X is None)
        x, y, names, label = load_dataset(name, n=n, seed=int(seed))
        out = self._install(x, y, names, label, is_new_dataset=is_new)
        if out.get("ok"):
            self.dataset_id = name
        return out

    def cmd_load_data(self, X=None, y=None, names=None, label="dataset") -> Dict[str, Any]:
        """For tests/scripts: install arrays directly (inputs in [0,1])."""
        return self._install(np.asarray(X, float), np.asarray(y, float),
                             list(names), str(label))

    def cmd_load_csv(self, text: str = "", name: str = "upload.csv",
                     target: Optional[str] = None,
                     n: Optional[int] = None) -> Dict[str, Any]:
        """Browser CSV import (T12). Parses with the vendored, dependency-free
        numeric-CSV reader (``gui3.csv_import`` — GUI3 stands alone): UTF-8, one
        header row, numeric cells, missing-row drop; infers the target column
        (``y``/``target``/last) unless one is given. Installs the raw-scale X —
        the fit's QuantileTransformer maps it into the model's [0,1] space, so
        scan/solo/σ̂ stay correct (engine keeps ``self._xt``). Large files are
        row-subsampled to the interactivity cap."""
        if self._fitting():
            return {"ok": False, "error": "fit running — cancel first"}
        from .csv_import import parse_numeric_csv, CsvError
        try:
            ds = parse_numeric_csv(str(text), name, target=target)
        except CsvError as exc:  # actionable (names the row/column at fault)
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        X = np.asarray(ds["X"], float)
        y = np.asarray(ds["y"], float)
        tgt = ds["target"]
        cap = int(n) if n else CSV_MAX_N
        total = None
        if len(y) > cap:
            idx = np.random.default_rng(42).permutation(len(y))[:cap]
            X, y, total = X[idx], y[idx], len(ds["y"])
        names = list(ds["feature_names"])
        size = f"N={len(y)}" + (f"/{total}" if total else "")
        label = f"{ds['name']} · {size} · D={X.shape[1]} · target={tgt}"
        out = self._install(X, y, names, label, is_new_dataset=True)
        if out.get("ok"):
            self.dataset_id = ""  # not a registry id
            out["header"] = list(ds["header"])
            out["target"] = tgt
            if ds["rows_dropped"]:
                self._emit({"event": "warn",
                            "message": f"CSV: dropped {ds['rows_dropped']} row(s) "
                                       f"with missing selected values"})
        return out

    def _install(self, x, y, names, label,
                 is_new_dataset: bool = True) -> Dict[str, Any]:
        if self._fitting():
            return {"ok": False, "error": "fit running — cancel first"}
        self._cancel_warm()
        # dataset-level diagnostics (fit-independent) — computed off-lock
        noise_mf = _model_free_sigma2(x, y)
        max_corr = _max_abs_corr(x, names)
        quality = _data_quality(x, names)
        with self._lock:
            names_changed = list(names) != list(self.names)
            fresh = is_new_dataset or names_changed
            self.X, self.y, self.names = x, y, names
            self.dataset_label = label
            if fresh:
                # a genuinely new dataset: per-channel K, mutes and admitted pairs
                # are keyed by variable, so they must reset; the basis returns to the
                # Legendre default (user request — basis is never carried across sets).
                self.k = {nm: DEFAULT_K for nm in names}
                self.muted = set()
                self.pairs = set()
                self.term_orders = {}   # BR-06 (keyed by variable name)
                self._prev_term_orders = None  # BR-13 rocker stash
                self.var_muted = set()  # BR-01 (keyed by variable name)
                self.pair_k2_map = {}   # BR-04 (keyed by variable-name pair)
                self.var_pairs = set()  # BR-05 (keyed by variable-name pair)
                self.pair_muted = set() # K₂=0 pair mutes
                self.var_k2h = VAR_K2H_DEFAULT
                self.order2 = "off"
                self._order2_top = None
                # default-fit d-rule (auto_order2): ≤4 vars → ALL pairs now
                # (tiny clique, defaults are legendre/K=3 at this point);
                # 5..10 → stage TOP once the first fit yields the Sᶠ ranking;
                # >10 → first-order only (compute-time rule, user-set).
                # When the INITIAL TRY planner is ON it owns the install-time
                # default instead (applied below, after AUTO-RESET).
                self._auto_order2_pending = False
                self._auto_plan_pending = False
                self._plan = None
                if self.plan_speed == "off" and self.auto_order2 \
                        and 2 <= len(names) <= ORDER2_AUTO_TOP_MAX_D:
                    if len(names) <= ORDER2_AUTO_ALL_MAX_D:
                        self.pairs = self._clique(list(names))
                        self.order2 = "all"
                    else:
                        self._auto_order2_pending = True
                self.basis = "legendre"
                if self.auto_reset:
                    # AUTO-RESET also returns the tuning knobs to defaults
                    self.lambda1 = LAMBDA1_DEFAULT
                    self.lambda_h = LAMBDA_H_DEFAULT
                    self.lambda2 = LAMBDA2_DEFAULT
                    self.pair_k2 = None
                    self.strategy = "auto"
                    self.hetero = False
                if self.plan_speed != "off":
                    # INITIAL TRY: plan the first shot from (N, D, Z) — sets
                    # the K faders; "all" pairs are admitted right away, a
                    # "screened" subset is STAGED until the first fit yields
                    # the Sᶠ ranking (order2_default event, like the d-rule)
                    plan = self._first_try_plan(len(y), len(names))
                    self.k = {nm: int(plan["k1"]) for nm in names}
                    if plan["n_pairs"]:
                        if plan["pair_mode"] == "all":
                            self.pairs = self._clique(list(names))
                            self.pair_k2 = int(plan["k2"])
                            self.order2 = "plan"
                        else:
                            self._auto_plan_pending = True
                    self._plan = plan
            else:
                # same dataset + variables (only N changed / a reload): preserve
                # the whole mix, just re-key K/mutes/pairs onto the (identical) names
                nameset = set(names)
                self.k = {nm: self.k.get(nm, DEFAULT_K) for nm in names}
                self.muted = {nm for nm in self.muted if nm in names}
                self.pairs = {p for p in self.pairs if p <= nameset}
                if self.order2 == "all":  # same variables → same clique
                    self.pairs = self._clique(self._unmuted_names())
            self.result = None
            self.view = {}
            self.scan = None
            self.gate = None
            self.verify = None
            self.residual = None
            self._residual_base_model = None
            if fresh:
                self.residual_cfg = {"family": "nystrom", "n_centers": None,
                                     "width": None, "lam": None,
                                     "kernel": None}
            self.noise_model_free = noise_mf
            self.max_corr = max_corr
            self.data_quality = quality
            self.loo_samples = []
            self._yhat = None
            self._xt = None
            self._active_names = []
            self.status = "READY"
            self.stale = True
            self.error = None
            self._epoch += 1
        if quality:
            cols = ", ".join(q["name"] for q in quality)
            self._emit({"event": "warn", "message":
                        f"data quality: {cols} discrete/heavily tied — the "
                        "quantile marginals cannot be uniform, so Sobol "
                        "shares for these channels are unreliable"})
        return {"ok": True}

    def cmd_set_k(self, name: str, k: int) -> Dict[str, Any]:
        if name not in self.names:
            return {"ok": False, "error": f"unknown channel {name}"}
        self.k[name] = max(K_MIN, min(int(k), K_MAX))
        self._last_touched = name
        self._mark_stale()
        return {"ok": True}

    def cmd_set_all_k(self, k: int) -> Dict[str, Any]:
        """Master fidelity: set EVERY channel's K fader at once (used by the
        per-basis fidelity sliders and the LINEAR preset). Clamped like
        ``cmd_set_k``; per-channel faders stay usable afterwards."""
        if not self.names:
            return {"ok": False, "error": "no dataset loaded"}
        k = max(K_MIN, min(int(k), K_MAX))
        self.k = {nm: k for nm in self.names}
        self._last_touched = None
        self._warn_order2_cols()
        self._mark_stale()
        return {"ok": True, "k": k}

    def cmd_set_mute(self, name: str, muted: bool) -> Dict[str, Any]:
        if name not in self.names:
            return {"ok": False, "error": f"unknown channel {name}"}
        (self.muted.add if muted else self.muted.discard)(name)
        if len(self.muted) >= len(self.names):
            self.muted.discard(name)
            return {"ok": False, "error": "cannot mute every channel"}
        # ALL/TOP track the active channels: mute shrinks the clique, unmute
        # regrows it. When the full clique no longer fits, degrade in the open:
        # ALL → TOP (re-budgeted, warned) → SELECT (pairs kept as they were).
        if self.order2 in ("all", "top"):
            was = self.order2
            act = self._unmuted_names()
            cap = (ORDER2_ALL_MAX_COLS if was == "all"
                   else self._order2_budget())
            if len(act) >= 2 and self._projected_order2_cols(act) <= cap:
                # the full clique fits this mode's budget → exact ALL
                self.pairs = self._clique(act)
                self.order2 = "all"
                self._order2_top = None
            elif self._recompute_order2_top():
                if was == "all":
                    t = self._order2_top or {}
                    self._emit({"event": "warn", "message":
                                f"unmuting {name} pushed the full 2nd-order "
                                f"clique past the budget — switched to TOP "
                                f"{t.get('m')} of {t.get('of')} channels "
                                "(first-order Sᶠ rank, heuristic)"})
            else:
                self.order2 = "select"
                self._order2_top = None
                self._emit({"event": "warn", "message":
                            f"unmuting {name} grows the 2nd-order clique past "
                            "the budget and no Sᶠ ranking is available — "
                            "dropped to SELECT, pairs kept as they were"})
        self._mark_stale()
        return {"ok": True}

    def cmd_set_term_order(self, name: str, order: str) -> Dict[str, Any]:
        """BR-06 order-selective membership for a channel (DEC-054): ``'both'``
        (default — first-order + interactions), ``'pair'`` (interactions only —
        NON-HIERARCHICAL, the first-order block is excluded and its df not
        spent), ``'marginal'`` (first-order only — drops every pair touching
        the variable), or ``'none'`` (mean-EXCLUDED: the variable leaves the
        MEAN model entirely — both orders — while its column stays an INPUT
        of the fit; ``variable_orders={j: []}`` backend semantics). With
        HETERO the column keeps feeding the Stage-D VARIANCE model
        (noise-only); on a CONSTANT fit it is in NEITHER model — the channel
        remains available to an attached COMPLEMENT (BR-12; before Ses06 this
        was held inert). 'none' still refuses a variance-muted channel (it
        would vanish from both models — that is MUTE). The all-'none' limit
        is the INTERCEPT-ONLY mean, the complement-only base (EXPLORATORY —
        hard-wired label in the view). Honesty labels are surfaced on the
        channel and in the term-structure summary."""
        if name not in self.names:
            return {"ok": False, "error": f"unknown channel {name}"}
        modes = {"both": [1, 2], "pair": [2], "marginal": [1], "none": []}
        if order not in modes:
            return {"ok": False,
                    "error": "order must be one of both / pair / marginal / none"}
        if order == "none":
            # BR-12 (Ses06): 'none' is legal on a CONSTANT fit too — the
            # channel carries NO mean term but STAYS an input of the fit, so
            # an attached COMPLEMENT can capture its structure (the BR-11
            # projector leaves unsolved structure available). With HETERO the
            # historical noise-only reading applies on top (variance kept).
            # The all-'none' limit is the INTERCEPT-ONLY mean — the
            # complement-only base (EXPLORATORY; the guard is gone).
            if name in self.var_muted:
                return {"ok": False, "error":
                        f"{name}'s variance is asserted flat (purple M) — noise-only "
                        "membership on top would remove it from both models "
                        "(that is MUTE). Unmute its variance first."}
        if order == "both":
            self.term_orders.pop(name, None)
        else:
            self.term_orders[name] = modes[order]
        self._mark_stale()
        return {"ok": True, "order": order}

    def cmd_set_intercept_only(self, on: bool = True) -> Dict[str, Any]:
        """BR-13: the 1st-INDIVIDUAL rocker's OFF state — stage BR-12's
        all-'none' limit in ONE step: every ACTIVE channel goes mean-EXCLUDED,
        so the next fit solves the INTERCEPT-ONLY base (f0) and an attached
        COMPLEMENT alone carries structure (EXPLORATORY — hard-wired label;
        the conventional-smooth-vs-structured comparison). Semantically
        identical to clicking every M to its third state: with HETERO the
        channels stay variance-tracked (noise-only); on a CONSTANT fit they
        are in neither model but remain COMPLEMENT inputs. Refuses on
        variance-muted channels (the per-channel guard, applied to all).
        ``on=False`` restores the membership map exactly as it was when the
        rocker staged it (a later per-channel M edit overrides the stash)."""
        if not self.names:
            return {"ok": False, "error": "no dataset loaded"}
        if on:
            active = [nm for nm in self.names if nm not in self.muted]
            if not active:
                return {"ok": False, "error": "no active channels"}
            vm = [nm for nm in active if nm in self.var_muted]
            if vm:
                return {"ok": False, "error":
                        ", ".join(vm) + ": variance asserted flat (purple M) "
                        "— mean-excluding on top would remove them from both "
                        "models (that is MUTE). Unmute their variance first."}
            self._prev_term_orders = {nm: list(o)
                                      for nm, o in self.term_orders.items()}
            for nm in active:
                self.term_orders[nm] = []
        else:
            prev = self._prev_term_orders or {}
            self.term_orders = {nm: list(o) for nm, o in prev.items()
                                if nm in self.names}
            self._prev_term_orders = None
        self._mark_stale()
        return {"ok": True, "intercept_only": bool(on)}

    def _intercept_only_staged(self) -> bool:
        """Settings-level BR-13 flag: every ACTIVE channel is mean-excluded
        (the staged complement-only base — the FITTED flag is
        ``view.master.intercept_only``)."""
        active = [nm for nm in self.names if nm not in self.muted]
        return bool(active) and all(self.term_orders.get(nm) == []
                                    for nm in active)

    def cmd_set_var_mute(self, name: str, muted: bool) -> Dict[str, Any]:
        """BR-01 variance-side mute (DEC-054): assert the noise is CONSTANT along
        ``name`` — drop it from the first-order variance model (Sʰ ≡ 0, df not
        spent) while keeping it in the mean model. Independent of the full MUTE.
        Only bites a heteroscedastic fit. Cannot leave the variance model empty;
        a fully (mean-)muted channel is already out of both models."""
        if name not in self.names:
            return {"ok": False, "error": f"unknown channel {name}"}
        if muted:
            if self.term_orders.get(name) == []:
                return {"ok": False, "error":
                        f"{name} is NOISE-ONLY (mean-excluded) — muting its "
                        "variance too would remove it from both models; use "
                        "MUTE, or restore its mean membership first"}
            # the variance model spans the ACTIVE mean vars; don't empty it
            active = [nm for nm in self.names if nm not in self.muted]
            remaining = [nm for nm in active
                         if nm not in self.var_muted and nm != name]
            if name in active and not remaining:
                return {"ok": False,
                        "error": "cannot assert every variable variance-flat "
                                 "(the variance model would be empty)"}
            self.var_muted.add(name)
        else:
            self.var_muted.discard(name)
        self._mark_stale()
        return {"ok": True}

    def cmd_set_lambda(self, which: str, value: float) -> Dict[str, Any]:
        value = float(value)
        if not (LAMBDA_MIN <= value <= LAMBDA_MAX):
            return {"ok": False, "error": "lambda out of range"}
        if which == "lambda1":
            self.lambda1 = value
        elif which == "lambda_h":
            self.lambda_h = value
        elif which == "lambda2":
            self.lambda2 = value
        else:
            return {"ok": False,
                    "error": f"unknown lambda {which} (wired: lambda1, lambda2, lambda_h)"}
        self._mark_stale()
        return {"ok": True}

    def cmd_set_pair_k(self, k: int) -> Dict[str, Any]:
        """Interaction fidelity: the harmonic order K2 of EVERY pair block. The
        backend takes a single scalar K2 for all pairs (no per-pair K2), so this is
        one global control, capped at ``PAIR_K2_CAP``. Only affects fits that have
        admitted interactions; a first-order fit ignores it."""
        k = int(k)
        if not (1 <= k <= PAIR_K2_CAP):
            return {"ok": False, "error": f"pair K must be 1..{PAIR_K2_CAP}"}
        self.pair_k2 = k
        self._warn_order2_cols()
        self._mark_stale()
        return {"ok": True}

    def cmd_set_pair_k2(self, a: str, b: str, k: int) -> Dict[str, Any]:
        """BR-04 per-pair interaction fidelity (DEC-053): set the harmonic order
        K2 of the ``a×b`` pair block independently. ``k`` in 1..PAIR_K2_CAP sets
        an override (and unmutes); ``k == 0`` MUTES the pair's interaction term
        (fader-at-zero — the component leaves the fit, the pair stays admitted
        in ROUTE and its strip stays visible); ``k < 0`` clears both mute and
        override (back to the global PAIRS-K). Any live override or mute uses
        the K2-mapping form, which PINS the exact admitted pairs (no
        clique-induced extras). Only affects fits with admitted interactions."""
        if a not in self.names or b not in self.names:
            return {"ok": False, "error": "unknown channel in pair"}
        if a == b:
            return {"ok": False, "error": "a pair needs two distinct channels"}
        key = self._pair_key(a, b)
        k = int(k)
        if k < 0:
            self.pair_k2_map.pop(key, None)
            self.pair_muted.discard(key)
        elif k == 0:
            if key not in self.pairs:
                return {"ok": False,
                        "error": f"{a}×{b} is clique-induced, not requested — "
                                 "mute a requested pair or reshape the clique "
                                 "in 2nd-order routing"}
            self.pair_muted.add(key)
        elif k > PAIR_K2_CAP:
            return {"ok": False, "error": f"pair K must be 0..{PAIR_K2_CAP} "
                                          "(0 mutes the pair term)"}
        else:
            self.pair_k2_map[key] = k
            self.pair_muted.discard(key)
        self._warn_order2_cols()
        self._mark_stale()
        return {"ok": True}

    def cmd_set_var_pair(self, a: str, b: str, on: bool = True) -> Dict[str, Any]:
        """BR-05 second-order variance term on the ``a×b`` interaction (DEC-053):
        assert the NOISE depends on this interaction (a per-pair Sʰ term). Only
        bites a heteroscedastic fit; both endpoints must be variance-active (an
        endpoint asserted variance-flat by BR-01 cannot carry a variance pair —
        the backend rejects it)."""
        if a not in self.names or b not in self.names:
            return {"ok": False, "error": "unknown channel in pair"}
        if a == b:
            return {"ok": False, "error": "a pair needs two distinct channels"}
        key = self._pair_key(a, b)
        if on:
            if a in self.var_muted or b in self.var_muted:
                return {"ok": False,
                        "error": "an endpoint is asserted variance-flat (purple M) — "
                                 "a variance pair cannot involve it"}
            # a K₂=0-MUTED pair may still carry a variance term: the mean
            # interaction and the noise interaction are independent choices
            # (the backend validates variance pairs against variance
            # variables, not the mean pair set)
            self.var_pairs.add(key)
        else:
            self.var_pairs.discard(key)
        self._mark_stale()
        return {"ok": True}

    def cmd_set_var_k2h(self, k: int) -> Dict[str, Any]:
        """BR-05 variance-pair order K2h (scalar, all variance pairs share it),
        capped at ``PAIR_K2_CAP``. Only affects a heteroscedastic fit that has
        variance pairs."""
        k = int(k)
        if not (1 <= k <= PAIR_K2_CAP):
            return {"ok": False, "error": f"variance pair order must be 1..{PAIR_K2_CAP}"}
        self.var_k2h = k
        self._mark_stale()
        return {"ok": True}

    def _pair_key(self, a: str, b: str) -> frozenset:
        return frozenset((str(a), str(b)))

    def _unmuted_names(self) -> List[str]:
        return [nm for nm in self.names if nm not in self.muted]

    def _clique(self, names: List[str]) -> set:
        """All pairs among ``names`` — exactly the clique the backend admits."""
        return {self._pair_key(a, b)
                for i, a in enumerate(names) for b in names[i + 1:]}

    def _sf_ranking(self) -> Optional[List[str]]:
        """Active channels ranked by the LAST fit's first-order Sᶠ (descending).
        None when no fit has produced shares yet, or every share is missing.
        Channels unmuted after that fit have no share and rank last — the TOP
        heuristic works from what the desk has actually measured, disclosed."""
        chs = (self.view or {}).get("channels") or []
        sf = {c["name"]: c.get("sf") for c in chs}
        act = self._unmuted_names()
        ranked = [nm for nm in act if sf.get(nm) is not None]
        if not ranked:
            return None
        ranked.sort(key=lambda nm: -float(sf[nm]))
        return ranked + [nm for nm in act if sf.get(nm) is None]

    def _order2_budget(self) -> int:
        """The quick-fit ('moving') order-2 column budget: interactivity cap
        AND a data-size guard (≤ N/2 order-2 columns), so small-N/high-d
        datasets get a small clique or an honest 'first-order only' answer
        while large-N/low-d datasets (e.g. ishigami) get the full clique."""
        n = 0 if self.y is None else int(len(self.y))
        return min(ORDER2_ALL_MAX_COLS, n // 2)

    def _recompute_order2_top(self) -> bool:
        """Re-derive the TOP budgeted clique after a mute/unmute. False when no
        Sᶠ ranking is available or even one pair exceeds the budget (the caller
        degrades to SELECT); True = pairs / order2 / _order2_top updated."""
        ranked = self._sf_ranking()
        act = self._unmuted_names()
        if ranked is None or len(act) < 2:
            return False
        budget = self._order2_budget()
        m = len(ranked)
        while m >= 2 and self._projected_order2_cols(ranked[:m]) > budget:
            m -= 1
        if m < 2:
            return False
        top = ranked[:m]
        self.pairs = self._clique(top)
        if m == len(act):  # nothing cut → exact ALL, no heuristic label
            self.order2 = "all"
            self._order2_top = None
        else:
            self.order2 = "top"
            self._order2_top = {"m": m, "of": len(act), "names": top}
        return True

    # ---------------------------------------------- INITIAL TRY (planner)
    def _basis_cols_fn(self):
        """Columns of ONE variable's first-order block at order k for the
        current basis — the planner's cost unit (a pair block costs the
        square). Lazy backend import (module import stays JAX-free)."""
        from hifi_anova.core.features import basis_size  # flagged import
        basis = self.basis
        return lambda k: int(basis_size(int(k), True, basis))

    def _first_try_plan(self, n: int, d: int) -> Dict[str, Any]:
        """Run the budget-ladder planner for the current speed preset, bounded
        by the desk's own caps (K_MAX, PAIR_K2_CAP)."""
        return plan_first_try(
            n, d, PLAN_Z[self.plan_speed], self._basis_cols_fn(),
            gamma=PLAN_GAMMA, min_pairs=PLAN_MIN_PAIRS,
            k1_max=min(5, K_MAX), k2_max=min(2, PAIR_K2_CAP))

    def _plan_pair_ranking(self) -> Optional[List[frozenset]]:
        """Candidate pairs over the active channels ranked by the heredity
        score Sᶠ(a)+Sᶠ(b) from the LAST fit (descending; name tie-break for
        determinism) — the INITIAL TRY screen. Excludes muted pairs. None
        when no fit has produced a ranking yet."""
        if self._sf_ranking() is None:
            return None
        chs = (self.view or {}).get("channels") or []
        sf = {c["name"]: float(c.get("sf") or 0.0) for c in chs}
        act = self._unmuted_names()
        cands = [self._pair_key(a, b)
                 for i, a in enumerate(act) for b in act[i + 1:]]
        cands = [p for p in cands if p not in self.pair_muted]
        cands.sort(key=lambda p: (-sum(sf.get(nm, 0.0) for nm in p),
                                  tuple(sorted(p))))
        return cands

    def _apply_plan_pairs(self, plan: Dict[str, Any]) -> bool:
        """Admit the plan's screened pair subset: the top-n_pairs candidates
        by heredity rank, PINNED exactly via the K2-mapping form (per-pair
        overrides at the planned K₂ — the clique closure would re-admit cut
        pairs). False when no ranking is available or nothing qualifies."""
        ranked = self._plan_pair_ranking()
        if not ranked:
            return False
        chosen = ranked[:int(plan["n_pairs"])]
        if not chosen:
            return False
        self.pairs = set(chosen)
        for p in chosen:  # user-set per-pair overrides stay theirs
            self.pair_k2_map.setdefault(p, int(plan["k2"]))
        self.order2 = "plan"
        self._order2_top = None
        self._plan = dict(plan)
        return True

    def _plan_quick_shot(self, act: List[str]) -> Dict[str, Any]:
        """Plan the '+2nd' quick shot for the CURRENT mix: the INITIAL TRY
        budget minus the first-order model the K faders already define is
        spent on pairs at a planned K₂ (bilinear first; upgraded to K₂=2 only
        when every pair fits and the per-D cap allows). The K faders are never
        touched here. An explicit user PAIRS-K is respected as the pair cost
        instead of the planned K₂. Returns the plan dict or ``{"error": …}``."""
        n = 0 if self.y is None else int(len(self.y))
        d = len(act)
        bc = self._basis_cols_fn()
        z = PLAN_Z[self.plan_speed]
        budget = min(z, n // PLAN_GAMMA)
        spent = 1 + sum(bc(int(self.k.get(nm, DEFAULT_K))) for nm in act)
        left = budget - spent
        all_pairs = d * (d - 1) // 2
        k2_user = int(self.pair_k2) if self.pair_k2 else None
        k2 = k2_user or 1
        per = bc(k2) ** 2
        take = min(all_pairs, max(0, left) // per)
        if take < min(PLAN_MIN_PAIRS, all_pairs):
            return {"error":
                    f"the INITIAL TRY budget (min(Z={z}, N/{PLAN_GAMMA}) = "
                    f"{budget} columns) is already spent by the first-order "
                    f"model (~{spent} columns) — lower K faders, pick a "
                    "slower preset"
                    + (f", lower PAIRS K (K₂={k2_user})" if k2_user else "")
                    + ", or use TOP/SELECT"}
        if take == all_pairs and k2_user is None:
            if default_k2_cap(d) >= 2 and PAIR_K2_CAP >= 2:
                extra = (bc(2) ** 2 - per) * take
                if left - take * per >= extra:
                    k2 = 2
        notes = []
        if take < all_pairs:
            notes.append(
                f"budget admits {take} of {all_pairs} pairs — ranked by the "
                "fitted first-order Sᶠ (heredity); a pair can matter with "
                "weak mains, SCAN covers the rest")
        return {"k2": int(k2), "n_pairs": int(take),
                "all_pairs": int(all_pairs),
                "pair_mode": "all" if take == all_pairs else "screened",
                "budget": int(budget),
                "cols": int(spent + take * bc(k2) ** 2),
                "binding": ("caps" if take == all_pairs
                            else ("speed" if z <= n // PLAN_GAMMA
                                  else "rows")),
                "notes": notes, "quick_shot": True,
                "k2_user": bool(k2_user)}

    def _projected_cols_admitted(self) -> Optional[int]:
        """Projected order-2 columns over the ADMITTED pair set (which under
        'plan'/'select' need not be a clique), honoring per-pair overrides and
        skipping muted pairs — same convention as _projected_order2_cols."""
        act = set(self._unmuted_names())
        prs = [p for p in self.pairs
               if set(p) <= act and p not in self.pair_muted]
        if not prs:
            return None
        from hifi_anova.core.features import basis_size  # flagged import
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in act] or [DEFAULT_K]
        base = self._effective_k2(ks)
        b2_of: Dict[int, int] = {}
        cols = 0
        for p in prs:
            kp = int(self.pair_k2_map.get(p) or base)
            if kp not in b2_of:
                b2_of[kp] = int(basis_size(kp, True, self.basis)) ** 2
            cols += b2_of[kp]
        return cols

    def _order2_snapshot_fields(self) -> Dict[str, Any]:
        """The order-2 settings block: mode, live projected column count over
        the mode's span, TOP preselection detail, moving budget min(cap, N/2),
        and the INITIAL TRY plan detail when the planner chose the structure."""
        span = self._order2_span()
        if self.order2 == "plan":
            cols = self._projected_cols_admitted()
        else:
            cols = (self._projected_order2_cols(span)
                    if span and len(span) >= 2 else None)
        return {
            "order2": self.order2,
            "order2_cols": cols,
            "order2_top": (dict(self._order2_top)
                           if self.order2 == "top" and self._order2_top
                           else None),
            "order2_budget": self._order2_budget(),
            # INITIAL TRY detail — present while the planner owns the mode OR
            # while a screened subset is staged awaiting the first ranking fit
            "order2_plan": (dict(self._plan)
                            if self._plan and (self.order2 == "plan"
                                               or self._auto_plan_pending)
                            else None),
            "plan_pending": bool(self._auto_plan_pending),
        }

    def _order2_span(self) -> Optional[List[str]]:
        """The channel set the current clique mode spans (ALL = all active,
        TOP = the stored preselection ∩ active), for column projections."""
        if self.order2 == "all":
            return self._unmuted_names()
        if self.order2 == "top" and self._order2_top:
            act = set(self._unmuted_names())
            return [nm for nm in self._order2_top["names"] if nm in act]
        return None

    def _projected_order2_cols(self, names: List[str]) -> int:
        """Projected order-2 column count Σ_pairs B₂(K₂ₚ)² for the clique over
        ``names`` at the current basis — the quantity the ALL-mode guardrail
        (and the EFFECTS readout) is stated in. Honors per-pair K₂ overrides
        and skips K₂=0-muted pairs, so the count tracks the pinned fit.
        Variance-pair (K2h) columns live in the Stage-D design and are not
        counted here."""
        if len(names) < 2:
            return 0
        from hifi_anova.core.features import basis_size  # flagged import
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in names]
        base = self._effective_k2(ks)
        b2_of = {}  # per distinct order (typically one or two)
        cols = 0
        for x in range(len(names)):
            for yv in range(x + 1, len(names)):
                key = self._pair_key(names[x], names[yv])
                if key in self.pair_muted:
                    continue
                kp = int(self.pair_k2_map.get(key) or base)
                if kp not in b2_of:
                    b2_of[kp] = int(basis_size(kp, True, self.basis)) ** 2
                cols += b2_of[kp]
        return cols

    def cmd_set_order2(self, mode: str) -> Dict[str, Any]:
        """EFFECTS (2nd ORDER): 'off' = first-order only (clears pairs),
        'all' = the full pair clique over the active channels (the backend's
        native admission — auto-tracks mute/unmute), 'top' = a BUDGETED clique
        over the top-m channels by the last fit's first-order Sᶠ (the largest m
        whose clique stays under ``ORDER2_ALL_MAX_COLS`` — an EXPLORATORY
        quick-fit heuristic, disclosed in the selection state; a channel that
        matters only at second order can be missed, SCAN covers the rest),
        'select' = hand-patched pairs via ROUTE (keeps the current set frozen).
        'all' is refused when the projected order-2 columns exceed the budget —
        with an actionable message (and ``too_big`` so clients can offer TOP),
        never a silently huge fit."""
        if mode not in ("off", "all", "top", "select", "plan"):
            return {"ok": False, "error":
                    f"unknown order-2 mode {mode} (off/all/top/select/plan)"}
        self._auto_order2_pending = False  # the user decided — no staged default
        self._auto_plan_pending = False
        if mode == "plan":
            # INITIAL TRY quick shot: spend what the budget leaves AFTER the
            # current K faders (never touches them) on pairs at a planned K₂
            if self.plan_speed == "off":
                return {"ok": False, "error":
                        "INITIAL TRY is OFF — pick a speed preset in the top "
                        "bar, or use ALL / TOP / SELECT"}
            act = self._unmuted_names()
            if len(act) < 2:
                return {"ok": False, "error": "INITIAL TRY needs ≥2 active channels"}
            shot = self._plan_quick_shot(act)
            if "error" in shot:
                return {"ok": False, "error": shot["error"]}
            if shot["pair_mode"] == "all" and shot.get("k2_user"):
                # the user's PAIRS-K governs and every pair fits — that IS
                # exact ALL; no heuristic label to carry
                return self.cmd_set_order2("all")
            if shot["pair_mode"] == "all":
                new = self._clique(act)
                changed = (new != self.pairs
                           or self.pair_k2 != int(shot["k2"]))
                self.pairs = new
                self.pair_k2 = int(shot["k2"])
                self.order2 = "plan"
                self._order2_top = None
                self._plan = shot
                if changed:
                    self._mark_stale()
            else:
                before = (set(self.pairs), dict(self.pair_k2_map))
                if not self._apply_plan_pairs(shot):
                    return {"ok": False, "error":
                            "INITIAL TRY screens pairs by the fitted "
                            "first-order Sᶠ (heredity) — run a first-order "
                            "FIT first"}
                if (set(self.pairs), dict(self.pair_k2_map)) != before:
                    self._mark_stale()
        elif mode == "all":
            act = self._unmuted_names()
            if len(act) < 2:
                return {"ok": False, "error": "ALL needs ≥2 active channels"}
            cols = self._projected_order2_cols(act)
            if cols > ORDER2_ALL_MAX_COLS:
                npairs = len(act) * (len(act) - 1) // 2
                return {"ok": False, "too_big": True, "error":
                        f"full 2nd order = {npairs} pairs ≈ {cols} order-2 "
                        f"columns (limit {ORDER2_ALL_MAX_COLS}) — too big to "
                        "stay interactive. Use TOP (budgeted clique over the "
                        "strongest channels), lower PAIRS K, mute channels, "
                        "or SELECT specific pairs"}
            new = self._clique(act)
            changed = new != self.pairs
            self.pairs = new
            self.order2 = "all"
            self._order2_top = None
            if changed:
                self._mark_stale()
        elif mode == "top":
            act = self._unmuted_names()
            if len(act) < 2:
                return {"ok": False, "error": "TOP needs ≥2 active channels"}
            budget = self._order2_budget()
            if self._projected_order2_cols(act) <= budget:
                # the FULL clique fits the moving budget — nothing to cut, so
                # this is exact ALL (no heuristic label, no ranking needed)
                return self.cmd_set_order2("all")
            ranked = self._sf_ranking()
            if ranked is None:
                return {"ok": False, "error":
                        "TOP ranks channels by the fitted first-order Sᶠ — "
                        "run a first-order FIT first"}
            m = len(ranked)
            while m >= 2 and (self._projected_order2_cols(ranked[:m])
                              > budget):
                m -= 1
            if m < 2:
                return {"ok": False, "error":
                        f"even a single pair exceeds the quick-fit budget "
                        f"(min({ORDER2_ALL_MAX_COLS}, N/2) = {budget} order-2 "
                        "columns) — first-order only is the honest quick fit "
                        "here; lower PAIRS K or SELECT one pair deliberately"}
            top = ranked[:m]
            new = self._clique(top)
            changed = new != self.pairs
            self.pairs = new
            self.order2 = "top"
            self._order2_top = {"m": m, "of": len(act), "names": top}
            if changed:
                self._mark_stale()
        elif mode == "select":
            # a workflow state: keeps whatever is patched (possibly nothing);
            # no structural change, so the fit is not marked stale
            self.order2 = "select"
            self._order2_top = None
        else:
            changed = bool(self.pairs)
            self.pairs = set()
            self.order2 = "off"
            self._order2_top = None
            if changed:
                self._mark_stale()
        return {"ok": True, "order2": self.order2,
                "pairs": self._pairs_as_lists()}

    def _sync_order2_after_manual_edit(self) -> None:
        """Manual pair edits (ROUTE click / set_pairs) own the set: a non-empty
        set is SELECT by definition; emptying it turns 2nd order off (an
        explicit ``set_order2('select')`` re-enters the workflow state)."""
        self.order2 = "select" if self.pairs else "off"
        self._order2_top = None
        self._auto_order2_pending = False  # manual edits override the default

    def _warn_order2_cols(self) -> None:
        """A knob change (basis / PAIRS K / K faders) under ALL/TOP can grow
        the clique's column count past the entry guardrail — allowed (the knob
        is the user's), but surfaced, never silent."""
        span = self._order2_span()
        if not span or len(span) < 2:
            return
        cols = self._projected_order2_cols(span)
        if cols > ORDER2_ALL_MAX_COLS:
            self._emit({"event": "warn", "message":
                        f"full 2nd order is now ~{cols} order-2 columns "
                        f"(> {ORDER2_ALL_MAX_COLS}) — expect a slow fit; lower "
                        "PAIRS K, mute channels, or switch 2nd order to SELECT"})

    def cmd_patch_pair(self, a: str, b: str, on: bool = True) -> Dict[str, Any]:
        """ROUTE (M3): admit (on) or remove (off) the interaction between two
        channels. The backend admits pairs only as an active-variable clique
        ('both' mode), so a single pair is exact; admitting pairs that don't share
        endpoints induces the clique closure — surfaced honestly at fit time, never
        silently reshaped. Marks the mix stale (a pair fit is a new compile shape)."""
        if a not in self.names or b not in self.names:
            return {"ok": False, "error": f"unknown channel(s) {a},{b}"}
        if a == b:
            return {"ok": False, "error": "a pair needs two distinct channels"}
        key = self._pair_key(a, b)
        (self.pairs.add if on else self.pairs.discard)(key)
        self._sync_order2_after_manual_edit()
        self._mark_stale()
        return {"ok": True, "pairs": self._pairs_as_lists()}

    def cmd_set_pairs(self, pairs=None) -> Dict[str, Any]:
        """Replace the admitted-interaction set wholesale (tests/scripts/clients
        that track the full list). ``pairs`` = iterable of [a, b] name pairs."""
        new: set = set()
        for pr in (pairs or []):
            if len(pr) != 2 or pr[0] == pr[1]:
                return {"ok": False, "error": f"bad pair {pr}"}
            a, b = str(pr[0]), str(pr[1])
            if a not in self.names or b not in self.names:
                return {"ok": False, "error": f"unknown channel in pair {pr}"}
            new.add(self._pair_key(a, b))
        self.pairs = new
        self._sync_order2_after_manual_edit()
        self._mark_stale()
        return {"ok": True, "pairs": self._pairs_as_lists()}

    def _pairs_as_lists(self) -> List[List[str]]:
        """The admitted-interaction set as a sorted, JSON-safe list of [a, b]."""
        return sorted(sorted(p) for p in self.pairs)

    def _effective_k2(self, ks: List[int]) -> int:
        """The single pair harmonic order used for ALL pair blocks. The backend
        takes one scalar K2 (no per-pair K2), so this is a global value: the user
        override ``self.pair_k2`` if set, else the max first-order K, always capped
        at ``PAIR_K2_CAP`` (pair columns grow as K2²)."""
        base = int(self.pair_k2) if self.pair_k2 else int(max(ks))
        return max(1, min(base, PAIR_K2_CAP))

    def _pair_surface(self, res, i: int, j: int, G: int):
        """The ORTHOGONAL second-order component f̂ᵢⱼ(x_i, x_j) over a G×G grid
        in the model's [0,1] space — the pair block's coefficients times its
        tensor basis (backend ``component_eval``), i.e. the pure Hoeffding
        interaction term with NO intercept and NO first-order effects in it
        (exactly what the Sᶠ₂ meter measures). NOT a ŷ slice: a predict-on-grid
        slice would superpose the marginals and hide the interaction shape.
        Shared by ``solo_pair`` (the SLICE surface) and the pair strip
        thumbnails so both show the same picture. Returns (g, z) with
        ``z[r][c] = f̂ᵢⱼ(x_i=g[c], x_j=g[r])`` (client convention: x_i on the
        x-axis/columns, x_j on the y-axis/rows)."""
        from hifi_anova.analysis.component_eval import second_order_on_grid
        model = res.model
        pidx = np.asarray(model.pair_indices).reshape(-1, 2)
        hits = [pk for pk, row in enumerate(pidx)
                if {int(row[0]), int(row[1])} == {i, j}]
        if not hits:
            raise ValueError(f"pair ({i},{j}) has no block in the fitted model")
        g, _, f = second_order_on_grid(model, hits[0], n_points=G)
        # second_order_on_grid uses 'ij' indexing (rows vary x_i) — transpose
        # to the client's rows-vary-x_j convention
        return np.asarray(g), np.asarray(f, float).T

    @staticmethod
    def _sym_zlim(z) -> float:
        """Color-scale limit for a SIGNED component surface: symmetric about 0
        (mid-ramp = f̂ᵢⱼ = 0), so positive/negative lobes read honestly."""
        zl = float(np.nanmax(np.abs(z)))
        return zl if np.isfinite(zl) and zl > 0.0 else 1e-12

    def cmd_set_basis(self, name: str) -> Dict[str, Any]:
        if name not in ("legendre", "fourier", "haar"):
            return {"ok": False, "error": f"unknown basis {name}"}
        self.basis = name
        self._warn_order2_cols()
        self._mark_stale()
        return {"ok": True}

    STRATEGIES = ("auto", "uniform", "variance", "smoothness", "curvature",
                  "sobolev", "spectral")

    def cmd_set_strategy(self, name: str) -> Dict[str, Any]:
        """Penalty shape (TONE) for the mean model. 'auto' keeps the backend
        default (variance for homo, curvature for hetero); sobolev/spectral
        use their default parameters (s=1, α=2). λ sliders set the amount,
        this sets the shape across basis frequencies — same-shape refit, so
        changing it never recompiles."""
        if name not in self.STRATEGIES:
            return {"ok": False, "error": f"unknown strategy {name} "
                                          f"(wired: {', '.join(self.STRATEGIES)})"}
        self.strategy = name
        self._mark_stale()
        return {"ok": True}

    def cmd_set_noise(self, hetero: bool) -> Dict[str, Any]:
        self.hetero = bool(hetero)
        self._mark_stale()
        return {"ok": True}

    def cmd_set_auto_reset(self, on: bool) -> Dict[str, Any]:
        """Toggle AUTO-RESET: when on, loading a NEW dataset returns the tuning
        knobs (λ, TONE, noise mode) to defaults. Per-channel K, mutes and the
        basis reset on a new dataset regardless; changing only N preserves the
        whole mix. Not a mix change → no stale."""
        self.auto_reset = bool(on)
        return {"ok": True}

    def cmd_set_auto_order2(self, on: bool) -> Dict[str, Any]:
        """Toggle the default-fit d-rule (ALL ≤4 vars / TOP 5..10 / first-order
        above) applied when a NEW dataset is installed. Affects future installs
        only — never restructures the current mix. Not a mix change → no stale."""
        self.auto_order2 = bool(on)
        if not self.auto_order2:
            self._auto_order2_pending = False
        return {"ok": True}

    def cmd_set_plan_speed(self, speed: str) -> Dict[str, Any]:
        """INITIAL TRY speed preset (off/fast/balanced/thorough). Governs the
        first-shot budget ladder at dataset install and the '+2nd' quick shot
        (order-2 mode 'plan'); 'off' restores the d-rule/TOP behavior. Never
        restructures the current mix by itself → no stale."""
        speed = str(speed).lower()
        if speed not in PLAN_SPEEDS:
            return {"ok": False, "error":
                    f"unknown INITIAL TRY speed {speed} "
                    f"({'/'.join(PLAN_SPEEDS)})"}
        self.plan_speed = speed
        if speed == "off":
            self._auto_plan_pending = False
        return {"ok": True, "plan_speed": self.plan_speed}

    def cmd_set_fit_backend(self, backend: str) -> Dict[str, Any]:
        """Array backend for fits and desk analytics: 'auto' (the numpy exact
        core — same code, no per-shape XLA compiles), 'numpy' (explicit), or
        'jax' (the previous behavior; per-shape compile costs return, warm
        neighbors were decided at server start). Statistically identical by
        construction (shared code; float32-level numeric differences only,
        recorded in the fit's provenance). Marks stale — the next fit runs on
        the chosen backend."""
        backend = str(backend).lower()
        if backend not in ("auto", "numpy", "jax"):
            return {"ok": False,
                    "error": f"unknown backend {backend} (auto/numpy/jax)"}
        if backend != self.fit_backend:
            self.fit_backend = backend
            self._mark_stale()
        return {"ok": True, "fit_backend": self.fit_backend}

    def cmd_reset_settings(self) -> Dict[str, Any]:
        """RESET ALL: return every mix setting to its default now (keeps the
        dataset)."""
        with self._lock:
            self.k = {nm: DEFAULT_K for nm in self.names}
            self.muted = set()
            self.pairs = set()
            self.order2 = "off"
            self._order2_top = None
            self._plan = None
            self._auto_plan_pending = False
            self.lambda1 = LAMBDA1_DEFAULT
            self.lambda_h = LAMBDA_H_DEFAULT
            self.lambda2 = LAMBDA2_DEFAULT
            self.pair_k2 = None
            self.pair_k2_map = {}
            self.term_orders = {}
            self._prev_term_orders = None
            self.var_muted = set()
            self.var_pairs = set()
            self.pair_muted = set()
            self.var_k2h = VAR_K2H_DEFAULT
            self.basis = "legendre"
            self.strategy = "auto"
            self.hetero = False
        self._mark_stale()
        return {"ok": True}

    def cmd_fit(self) -> Dict[str, Any]:
        with self._lock:
            if self.X is None:
                return {"ok": False, "error": "no dataset loaded"}
            if self._active_job == "fit" or self._fit_queued:
                return {"ok": False, "error": "fit already running", "busy": True}
            # a warm/verify/lootest/residual job may hold the worker — cancel it
            # so the fit starts asap (cooperative; a user fit outranks all)
            dropped = self._drain_jobs({"warm", "verify", "lootest",
                                        "residual"})
            self._verify_pending = False
            self._loo_test_pending = False
            self._residual_pending = False
            # a QUEUED complement never reaches the worker, so the worker's
            # discarded emission can't cover it — terminal event from the
            # cmd path (flag already settled above; plain _emit is correct
            # here, _emit_done only flushes when some job ends)
            if "residual" in dropped:
                self._emit({"event": "residual_discarded",
                            "reason": "superseded by fit"})
            if self._active_job in ("warm", "verify", "lootest", "residual"):
                self._stop.set()
            self._fit_queued = True
            self.status = "FITTING"
            self.error = None
            self._jobs.put(("fit", None))
            self._ensure_worker()
        return {"ok": True}

    def cmd_cancel(self) -> Dict[str, Any]:
        self._stop.set()
        dropped = self._drain_jobs({"fit", "warm", "verify", "lootest",
                                    "residual"})
        self._verify_pending = False
        self._loo_test_pending = False
        self._residual_pending = False
        if "residual" in dropped:   # queued job dropped — worker never saw it
            self._emit({"event": "residual_discarded", "reason": "cancelled"})
        if self._fit_queued and "fit" in dropped:
            with self._lock:
                self._fit_queued = False
                self.status = "FIT_READY" if self.result is not None else "READY"
            self._emit({"event": "cancelled"})
        return {"ok": True}

    def cmd_solo(self, name: str) -> Dict[str, Any]:
        with self._lock:
            res = self.result
            active = list(self._active_names)
            yhat = self._yhat
            xt = self._xt
        if res is None or name not in active:
            return {"ok": False, "error": f"no fitted channel {name}"}
        j = active.index(name)
        # component curves live on the model's [0,1] (quantile) scale —
        # scatter x must be in the same space (identity for uniform demos)
        cx, cy = res.component_curve(name, n_points=121)
        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        xcol = xt[:, j]
        partial = self.y - yhat + np.interp(xcol, cx, cy)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(xcol))[:500]
        payload: Dict[str, Any] = {
            "ok": True, "name": name, "xscale": "quantile",
            "curve": [[_f(v) for v in cx], [_f(v) for v in cy]],
            "scatter": [[_f(v) for v in xcol[idx]], [_f(v) for v in partial[idx]]],
        }
        if self.hetero:
            grid_t = np.full((len(cx), xt.shape[1]), 0.5)
            grid_t[:, j] = cx
            try:
                tr = res.transformer
                grid = (tr.inverse_transform(grid_t)
                        if tr is not None and hasattr(tr, "inverse_transform")
                        else grid_t)
                sig2 = np.asarray(res.sigma_x2(grid), float)
                payload["sigma"] = [_f(v) for v in np.sqrt(np.maximum(sig2, 0))]
            except Exception:
                payload["sigma"] = None
        return payload

    def cmd_solo_pair(self, a: str, b: str) -> Dict[str, Any]:
        """Orthogonal interaction component for an admitted pair (C15 SOLO →
        SLICE): the pure second-order term f̂ₐᵦ over a grid on (x_a, x_b) in the
        model's [0,1] space — see ``_pair_surface``. zmin/zmax are symmetric
        about 0 (signed component → zero-centered color scale)."""
        with self._lock:
            res = self.result
            active = list(self._active_names)
        if res is None or a not in active or b not in active:
            return {"ok": False, "error": f"no fitted pair {a}·{b}"}
        if self._pair_key(a, b) in self.pair_muted:
            return {"ok": False, "error": f"{a}×{b} is muted (K₂ at 0) — "
                                          "raise its K₂ to draw the component"}
        i, j = active.index(a), active.index(b)
        try:
            g, z = self._pair_surface(res, i, j, G=24)
        except Exception as exc:
            return {"ok": False, "error": f"surface failed: {type(exc).__name__}: {exc}"}
        zl = self._sym_zlim(z)
        return {"ok": True, "a": a, "b": b, "xscale": "quantile",
                "g": [_f(v) for v in g],
                "z": [[_f(v) for v in row] for row in z],
                "zmin": _f(-zl), "zmax": _f(zl)}

    def cmd_probe(self, x=None) -> Dict[str, Any]:
        """Prediction probe (PLAY): evaluate the CURRENT fit at one point.

        ``x`` maps active channel names to positions in the model's [0,1]
        quantile space (the SLICE convention; missing channels default to
        0.5). Returns ŷ, the 95% prediction interval (aleatoric + epistemic,
        Student-t — the public ``predict_intervals``), σ̂ at the point, and
        the per-variable first-order contributions f̂ᵢ(xᵢ) (the backend
        ``prediction_summary`` breakdown). One-point predicts are
        milliseconds, so this runs inline — no job queue."""
        with self._lock:
            res = self.result
            active = list(self._active_names)
        if res is None or not active:
            return {"ok": False, "error": "no fit to probe — FIT first"}
        if self._fitting():
            return {"ok": False, "error": "fit running — probe after it finishes"}
        x = x or {}
        qpos = np.full(len(active), 0.5)
        for nm, v in x.items():
            if nm not in active:
                return {"ok": False, "error": f"unknown/muted channel {nm}"}
            qpos[active.index(nm)] = min(max(float(v), 0.0), 1.0)
        row_t = qpos[None, :]
        tr = getattr(res, "transformer", None)
        row = (tr.inverse_transform(row_t)
               if tr is not None and hasattr(tr, "inverse_transform") else row_t)
        try:
            yhat = float(np.asarray(res.predict(row), float).reshape(-1)[0])
            lo, hi = res.predict_intervals(row, alpha=0.05)
            lo, hi = float(np.asarray(lo).reshape(-1)[0]), \
                float(np.asarray(hi).reshape(-1)[0])
            if self.hetero:
                s2 = float(np.asarray(res.sigma_x2(row), float).reshape(-1)[0])
                sigma = float(np.sqrt(max(s2, 0.0)))
                sigma_kind = "pointwise"
            else:
                sigma = _f(res.sigma_hat)
                sigma_kind = "global"
            # per-variable first-order contributions at the point (the
            # prediction_summary breakdown; pair/residual terms are part of ŷ
            # but not split here — the client says so)
            from hifi_anova.analysis.component_eval import (  # flagged import
                evaluate_all_first_order)
            comps = evaluate_all_first_order(res.model, row_t)
            contributions = {nm: _f(float(np.asarray(comps[j]).reshape(-1)[0]))
                             for j, nm in enumerate(active)}
            try:
                f0 = _f(float(np.asarray(res.model.f0).reshape(-1)[0]))
            except Exception:
                f0 = None
            # C16: an attached RESIDUAL contributes to ŷ as its OWN line —
            # a smooth catch-all has no per-variable split (PROBE stays
            # first-order per variable; the client labels it separately)
            residual_c = None
            if getattr(res.model, "residual_net", None) is not None:
                try:
                    from hifi_anova.model.linear_residual import (
                        predict_residual_batch)
                    residual_c = _f(float(np.asarray(predict_residual_batch(
                        res.model.residual_net, row_t)).reshape(-1)[0]))
                except Exception:
                    residual_c = None
        except Exception as exc:
            return {"ok": False,
                    "error": f"probe failed: {type(exc).__name__}: {exc}"}
        return {"ok": True, "stale": bool(self.stale), "alpha": 0.05,
                "residual_contribution": residual_c,
                "x_q": {nm: _f(qpos[j]) for j, nm in enumerate(active)},
                "x_raw": {nm: _f(np.asarray(row, float)[0][j])
                          for j, nm in enumerate(active)},
                "yhat": _f(yhat), "lo": _f(lo), "hi": _f(hi),
                "sigma": sigma if sigma is None else _f(sigma),
                "sigma_kind": sigma_kind,
                "f0": f0, "contributions": contributions}

    def cmd_scan_pairs(self) -> Dict[str, Any]:
        """Routing scan: residual capture per unselected pair. Runs on the
        fitted model; demo inputs are already uniform (the model's training
        space) — CSV data will need the transformer applied here."""
        with self._lock:
            res = self.result
            active = list(self._active_names)
        if res is None:
            return {"ok": False, "error": "fit first"}
        if self._fitting():
            return {"ok": False, "error": "fit running"}
        from hifi_anova.analysis.interaction_discovery import scan_missing_pairs
        sc = scan_missing_pairs(res.model, self._xt, self.y, verbose=False)
        scores = {f"{active[i]}|{active[j]}": _f(v)
                  for (i, j), v in sc.pair_scores.items()}
        self.scan = {
            "scores": scores,
            "threshold": _f(getattr(sc, "flag_threshold",
                                    getattr(sc, "significance_threshold", 0.0))),
            "ranked": [[f"{active[i]}|{active[j]}", _f(v)]
                       for (i, j), v in sc.ranked_pairs[:10]],
        }
        return {"ok": True, **self.scan}

    def cmd_gate(self) -> Dict[str, Any]:
        """GATE (R7/C10): group-lasso L1 channel selection on the fitted
        first-order design. Unlike the λ sliders (every TONE is an L2 ridge),
        this zeroes WHOLE channels — the Gram-weighted group lasso
        ``γ Σ_g √df_g ||w_g||_G`` swept over γ with BIC. Returns the min-BIC
        active set plus each channel's entry order (the γ at which it first
        becomes active), so the desk can offer a parsimony dial and light gate
        LEDs. Applying a gate = muting the dropped channels (a real refit).

        DEC-045: variable selection is unsupported on a *mixed per-variable K*
        design (the group blocks have different sizes/Grams) — GATE reports
        ``mixed: True`` and the desk disables the control honestly."""
        with self._lock:
            res = self.result
            active = list(self._active_names)
        if res is None:
            return {"ok": False, "error": "fit first"}
        if self._fitting():
            return {"ok": False, "error": "fit running"}
        rec = getattr(res, "_fitted_design", None)
        if rec is None:
            return {"ok": False, "error": "no fitted design"}
        if getattr(rec, "sample_weights", None) is not None:
            return {"ok": False, "error": "GATE is a homoscedastic selection "
                                          "(Stage-D fit is GLS-weighted)"}
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in active]
        if rec.sobol_groups is not None or len(set(ks)) > 1:
            self.gate = {"mixed": True, "names": active}
            return {"ok": True, "mixed": True, "names": active,
                    "error": "selection unsupported on mixed per-variable K "
                             "(DEC-045) — set a uniform K to gate"}
        D = len(active)
        if D < 2:
            return {"ok": False, "error": "need ≥2 active channels to gate"}
        b1 = rec.block(1)
        if b1 is None:
            return {"ok": False, "error": "no first-order block"}
        from hifi_anova.training.selection import select_variables_glasso
        Phi1 = np.asarray(rec.Phi[:, b1.columns], float)
        yc = np.asarray(rec.y_centered, float)
        reg1 = np.asarray(rec.reg_diag[b1.columns], float)
        G1 = np.asarray(b1.gram, float)
        selected, info = select_variables_glasso(
            Phi1, yc, D, int(b1.K), reg1, G1=G1,
            include_linear=bool(b1.include_linear),
            basis_name=str(b1.basis_name), verbose=False)
        # entry γ per group: the largest γ at which the group is active
        entry: Dict[int, float] = {}
        for p in info.get("path", []):
            for g in p.get("active_groups", []):
                if g not in entry:
                    entry[g] = float(p["gamma"])
        # order channels by entry (earliest entry = strongest → kept first)
        order = sorted(range(D), key=lambda g: -entry.get(g, 0.0))
        self.gate = {
            "mixed": False,
            "names": active,
            "order": [active[g] for g in order],
            "selected": sorted(active[g] for g in selected),
            "n_bic": len(selected),
            "d": D,
            "entry_log10": {active[g]: _f(np.log10(entry[g]))
                            for g in entry if entry[g] > 0},
        }
        return {"ok": True, **self.gate}

    # ------------------------------------------------------------- selection (R16)
    def _selection_state(self) -> Dict[str, Any]:
        """CONFIG-CONDITIONAL honesty (R16): which in-session decisions selected the
        model's *structure* on the SAME data being interpreted. Derived from live
        engine state (self-correcting — un-muting / un-patching clears it), so the
        lamp reflects the actual mix, not a static reminder. Both mutes (variable
        selection, whether by hand or via GATE APPLY) and admitted interactions
        (chosen off SCAN / this fit) make the reported shares exploratory."""
        reasons: List[str] = []
        if (self.order2 == "plan" and self._plan
                and self._plan.get("pair_mode") == "screened"):
            p = self._plan
            reasons.append(
                f"2nd-order pairs screened by the INITIAL TRY budget ladder "
                f"({p['n_pairs']} of {p['all_pairs']} pairs, Sᶠ-heredity "
                "rank — heuristic first shot)")
        if self.order2 == "top" and self._order2_top:
            t = self._order2_top
            reasons.append(
                f"2nd-order clique preselected by first-order Sᶠ rank "
                f"(TOP {t['m']} of {t['of']} channels — heuristic quick fit)")
        elif self.pairs and not reasons:
            pl = ["×".join(sorted(p)) for p in
                  sorted(self.pairs, key=lambda s: tuple(sorted(s)))]
            reasons.append("interactions patched (" + ", ".join(pl) + ")")
        if self.muted:
            reasons.append("variables muted/selected (" +
                           ", ".join(sorted(self.muted)) + ")")
        return {"active": bool(reasons), "reasons": reasons}

    def _noise_triangulation(self) -> Optional[Dict[str, Any]]:
        """Agreement verdict across the three noise-floor legs (RSS/df, min of
        the σ²(λ₁) noise-complexity curve, replicate/Rice model-free). The
        spread = max/min of the available legs; ``agree`` applies the
        ``NOISE_AGREE_SPREAD`` threshold — computed HERE (it was a JS literal)
        so the verdict is a tested engine claim. ``None`` with <2 legs."""
        m = (self.view or {}).get("master") or {}
        legs: List[Tuple[str, float]] = []
        sh = m.get("sigma_hat")
        if sh is not None and not m.get("sigma_is_calibration"):
            legs.append(("fit", float(sh) ** 2))
        p = (self.view or {}).get("path") or {}
        if p and p.get("sigma2_min") is not None:
            legs.append(("sigma2_lambda_min", float(p["sigma2_min"])))
        mf = self.noise_model_free or {}
        if mf.get("sigma2") is not None:
            legs.append((str(mf.get("method") or "model_free"),
                         float(mf["sigma2"])))
        legs = [(nm, v) for nm, v in legs if np.isfinite(v) and v > 0]
        if len(legs) < 2:
            return None
        vals = [v for _, v in legs]
        spread = max(vals) / min(vals)
        return {"legs": [[nm, _f(v)] for nm, v in legs],
                "spread": _f(spread),
                "agree": bool(spread < NOISE_AGREE_SPREAD)}

    # ------------------------------------------------------------- report (PRINT, T9)
    def cmd_export_report(self) -> Dict[str, Any]:
        """PRINT (T9): a human-readable report + a runnable reproduction script for
        the CURRENT fit. The report is built from the live view/master (the same
        numbers the desk shows); ``summary_text`` captures the backend's own
        ``res.summary()`` (the authoritative honesty narrative — LOO tier, inference
        provenance, fidelity, two-fit gap); the script rebuilds the exact
        ``hifi_anova(...)`` call from the real fit kwargs so the fit reproduces."""
        if self.result is None or not self.view:
            return {"ok": False, "error": "no fit to report — FIT first"}
        res = self.result
        nms = list(self._active_names)
        # authoritative backend narrative (prints unicode; capture it verbatim)
        import contextlib
        import io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                res.summary()
            summary_text = buf.getvalue()
        except Exception as exc:  # never fail the export on a summary hiccup
            summary_text = f"(summary unavailable: {type(exc).__name__}: {exc})"
        report = self._report_markdown(nms)
        script = self._repro_script(nms)
        return {"ok": True, "report": report, "script": script,
                "summary_text": summary_text}

    def _report_markdown(self, nms: List[str]) -> str:
        m = (self.view or {}).get("master", {}) or {}
        chans = [c for c in (self.view or {}).get("channels", []) if not c.get("muted")]
        chans = sorted(chans, key=lambda c: -(c.get("sf") or 0.0))
        pairs = (self.view or {}).get("pairs", []) or []
        sel = self._selection_state()

        def g(v, d=4):
            return "—" if v is None else f"{float(v):.{d}f}"
        L = [f"# HiFi Console report — {self.dataset_label or 'dataset'}",
             "",
             f"- Dataset: **{self.dataset_label}** (`{self.dataset_id or 'custom'}`), "
             f"N = {0 if self.y is None else len(self.y)}, D = {len(self.names)}",
             f"- Basis: **{self.basis}**, TONE (penalty shape): **{self.strategy}**, "
             f"noise model: **{'HETERO (Stage-D)' if self.hetero else 'CONSTANT'}**",
             f"- λ₁ = {self.lambda1:.3e}"
             + (f", λ_h = {self.lambda_h:.3e}" if self.hetero else "")
             + (f", λ₂ = {self.lambda2:.3e}" if self.pairs else ""),
             ""]
        L += ["## Fit quality",
              f"- R² = {g(m.get('r2'))}  |  classical R² = {g(m.get('r2_classical'))}",
              (f"- LOO-R² = {g(m.get('loo_r2'))}" if m.get('loo_r2') is not None
               else "- LOO-R² = — (whitened after Stage-D; use LOO-NLL)"),
              f"- LOO-NLL = {g(m.get('loo_nll'), 3)}  ·  LOO tier {m.get('loo_tier')}",
              f"- effective df = {g(m.get('df'), 1)}  ·  𝓕 fidelity = {g(m.get('fidelity'))}",
              f"- {'σ̂ (CAL, whitened)' if m.get('sigma_is_calibration') else 'σ̂'} = "
              f"{g(m.get('sigma_hat'), 3)}",
              ""]
        # Ses08 (watch-list #2): the λ-bank criteria of THIS design (R28) belong
        # in the report — distinct from GATE's group-lasso BIC path.
        if m.get("aic") is not None:
            L += ["## Information criteria (this design's λ bank)",
                  f"- AIC = {g(m.get('aic'), 1)}  ·  BIC = {g(m.get('bic'), 1)}"
                  f"  ·  GCV = {g(m.get('gcv'), 5)}"
                  f"  ·  N/df = {g(m.get('ess_per_param'), 1)}",
                  ""]
        L += ["## First-order attribution (core Sobol Sᶠ)"]
        for c in chans:
            ci = c.get("ci")
            cis = f" [{g(ci[0],3)}, {g(ci[1],3)}]" if ci and ci[0] is not None else ""
            L.append(f"- `{c['name']}` (K={c.get('k')}): Sᶠ = {g(c.get('sf'),3)}{cis}"
                     + (f"  ·  Sʰ = {g(c.get('sh'),3)}" if c.get('sh') is not None else ""))
        if self.muted:
            L.append(f"- muted (excluded): {', '.join(sorted(self.muted))}")
        if pairs:
            L += ["", "## Interactions (order-2 Sᶠ)"]
            for p in pairs:
                tag = "" if p.get("requested") else " (induced by clique closure)"
                L.append(f"- `{p['a']} × {p['b']}`: Sᶠ = {g(p.get('sf'),3)}{tag}")
        gap = m.get("sobol_gap_max")
        L += ["", "## Honesty",
              f"- LOO tier {m.get('loo_tier')}"
              + ("" if m.get('tier2_ok') is None
                 else (" · Tier-II guarantee " + ("holds" if m.get('tier2_ok') else "AT RISK"))),
              ("- Findings are **EXPLORATORY** — " + "; ".join(sel["reasons"])
               + " (structure chosen on this same data; no post-selection coverage)")
              if sel["active"] else
              "- No in-session variable/interaction selection recorded on this fit.",
              ]
        mc = m.get("max_corr")
        if mc and mc.get("value") is not None:
            L.append(f"- Max |input corr| = {g(mc['value'],2)} ({'·'.join(map(str,mc['pair']))}, "
                     f"model space){' — shares are conditional' if mc['value']>=CORR_WARN else ''}")
        if gap and gap.get("value") is not None:
            L.append(f"- Two-fit gap (efficient−interpretable) max = {gap['value']:+.3f} "
                     f"on `{gap['name']}` — the predictive fit attributes differently.")
        # Ses08 (watch-list #2): a VERIFY oracle run / ΔLOO TEST that certifies
        # or ranks THIS fit belongs in its report (both are cleared on stale, so
        # if they exist here they describe exactly this fit).
        if self.verify:
            v = self.verify
            L += ["", "## VERIFY LOO (Tier-III oracle — this fit)",
                  f"- Oracle LOO-NLL = {g(v.get('loo_nll_oracle'), 4)} vs reported "
                  f"Tier-{v.get('tier_reported')} {g(v.get('loo_nll_reported'), 4)} "
                  f"(Δ = {'—' if v.get('delta') is None else format(v['delta'], '+.4f')}; "
                  f"{v.get('n_folds')} exact refits, {v.get('seconds')} s). "
                  "The oracle value is the authority."]
        if self.residual:
            rb = self.residual
            wp = rb.get("width_profile")
            kn = f", kernel = {rb['kernel']}" if rb.get("kernel") else ""
            L += ["", "## COMPLEMENT bus (EXPLORATORY smooth catch-all — this fit)",
                  f"- Family **{rb['family']}**{kn} (M = {rb['m']}, width = "
                  f"{g(rb.get('width'), 3)} "
                  f"{'user-set' if rb.get('width_source') == 'user' else ('auto — criterion-min over a disclosed width grid (' + wp.get('grid_note', '') + ')' if wp else 'auto — library default')}"
                  f"), λ_res = {rb['lambda']:.3e} "
                  f"({'user-set' if rb.get('lambda_source') == 'user' else 'auto'}"
                  f" — {rb.get('criterion')}; first guesses, overridable).",
                  "- Orthogonal in-sample: the structured coefficients and the "
                  "Sobol attribution above are unchanged by construction; the "
                  "population decomposition may shift.",
                  f"- Captured share (in-sample) = {g(rb.get('captured_share'), 3)}"
                  + (f"  ·  ΔLOO-R² (combined vs base) = "
                     f"{format(rb['delta_loo_r2'], '+.4f')} ± "
                     f"{g(rb.get('delta_loo_r2_se'))}"
                     if rb.get("delta_loo_r2") is not None else
                     "  ·  ΔLOO-R² unavailable (see notes)"),
                  f"- Combined df = {g(rb.get('df_combined'), 1)} "
                  f"(residual block {g(rb.get('df_block'), 1)})"]
            sb = rb.get("sobol")
            if sb:
                def _sline(st, se):
                    return "  ·  ".join(
                        f"{nm} {v:.3f}±{s:.3f}" if v is not None else f"{nm} —"
                        for nm, v, s in zip(sb["names"], st, se))
                L += ["- Sobol 'rides on' readout (total-order, "
                      f"{sb['estimator']}, n = {sb['n_qmc']}; {sb['label']}):",
                      f"  - S_T(ĝ) within Var(ĝ): {_sline(sb['st_g'], sb['st_g_se'])}",
                      f"  - S_T(f̂ = mean+ĝ) of model Var(f̂): "
                      f"{_sline(sb['st_f'], sb['st_f_se'])}",
                      f"  - Var(ĝ)/Var(f̂) = {g(sb.get('g_share_of_f'), 3)} "
                      "(scales the ĝ row onto total variance — BOTH "
                      "normalizations shown, neither blessed, §6.8).",
                      f"  - max S_1(ĝ) = "
                      f"{g(max((v for v in sb['s1_g'] if v is not None), default=None), 3)}"
                      " — per-variable orthogonality-defect diagnostic "
                      "(in-sample projection vs the uniform measure; "
                      "a positive value is expected, not a bug)."]
            ng = rb.get("noise_guard")
            if ng:
                L.append("- Noise-floor guard: implied σ̂² = "
                         f"{g(rb.get('sigma2_combined'), 4)} vs model-free "
                         f"({ng.get('method')}) {g(ng.get('floor_sigma2'), 4)}"
                         + (" — **BELOW the floor: likely fitting noise**"
                            if ng.get("triggered") else " — above the floor"))
            for note in rb.get("notes") or []:
                L.append(f"- ⚠ {note}")
        if self.loo_test and self.loo_test.get("rows"):
            lt = self.loo_test
            L += ["", "## ΔLOO TEST (rank-only, EXPLORATORY — this fit)",
                  f"- {lt['n_candidates']} candidate refits vs base LOO-R² "
                  f"{g(lt.get('base_loo_r2'))}; paired per-point SE; "
                  "NO keep/drop rule (stopping rule awaits expert sign-off)."]
            for r in lt["rows"][:5]:
                if "error" in r:
                    L.append(f"- `{r['label']}`: {r['error']}")
                else:
                    d, dd = r.get("delta_loo_r2"), r.get("delta_df")
                    L.append(f"- `{r['label']}`: ΔLOO-R² = "
                             f"{'—' if d is None else format(d, '+.4f')} "
                             f"± {g(r.get('se_loo_r2'))} "
                             f"(Δdf {'—' if dd is None else format(dd, '+.1f')})"
                             + (f" — {r['note']}" if r.get("note") else ""))
            for n in lt.get("notes") or []:
                L.append(f"- ⚠ {n}")
        L += ["", "---", "*Generated by the HiFi Console (GUI3). Selection here is "
              "exploratory; see the reproduction script for the exact fit call.*"]
        return "\n".join(L)

    def _repro_script(self, nms: List[str]) -> str:
        """A runnable Python snippet that reconstructs the exact fit. Uses the kwargs
        the last fit ACTUALLY ran with (so a mixed-K→uniform backend fallback is
        reflected, not re-derived). For a built-in registry dataset it also emits the
        loader used by the console; a CSV / custom dataset gives a concrete pandas
        template (the data is the user's, so X, y stay a marked placeholder)."""
        # prefer the recorded actual kwargs; fall back to a fresh derivation only if
        # (somehow) unavailable
        if self._last_fit_kwargs is not None:
            kwargs = dict(self._last_fit_kwargs)
            active = list(self._last_active or
                          [i for i, nm in enumerate(self.names) if nm not in self.muted])
        else:
            active = [i for i, nm in enumerate(self.names) if nm not in self.muted]
            ks = [int(self.k.get(nm, DEFAULT_K)) for nm in nms]
            _, endpoints = self._active_pairs(nms)
            kwargs = self._fit_kwargs(ks, nms, self.hetero, pairs=endpoints or None)
        # drop non-serialisable / runtime handles from the display kwargs
        for drop in ("should_stop", "progress", "verbose", "seed"):
            kwargs.pop(drop, None)
        # C16: an attached RESIDUAL reproduces as the pipeline's own Stage C —
        # the same construction (same seed → same split, fixed λ_res) run
        # in-pipeline instead of post-hoc (DEC-057 routes it to the core).
        if self.residual:
            rb = self.residual
            m_key, w_key, _ = RESIDUAL_PARAM_KEYS[rb["family"]]
            kwargs["residual"] = rb["family"]
            kwargs["residual_config"] = {m_key: int(rb["m"]),
                                         w_key: float(rb["width"])}
            if rb.get("kernel"):               # nystrom Matérn ν (Ses04)
                kwargs["residual_config"]["kernel"] = str(rb["kernel"])
            kwargs["lambda_residual"] = float(rb["lambda"])
            # ordering caveat: the pipeline runs Stage C BEFORE Stage D,
            # while the console fitted this residual post-hoc on the
            # Stage-D-adapted mean's leftover — same construction, but on a
            # hetero fit the weights are NOT byte-identical (surfaced below)

        def render_val(v):
            if isinstance(v, str):
                return repr(v)
            if isinstance(v, bool):
                return "True" if v else "False"
            if isinstance(v, float):
                return f"{v:.6g}"
            return repr(v)
        kw_lines = ",\n".join(f"    {k}={render_val(v)}" for k, v in kwargs.items())
        L = ["import numpy as np",
             "import hifi_anova as ha",
             ""]
        registry_ids = {d["id"] for d in REGISTRY} if REGISTRY else set()
        if self.dataset_id in registry_ids:
            L += [f"# data: built-in registry dataset '{self.dataset_id}' as loaded by the console",
                  "from gui3.engine import ConsoleEngine",
                  "eng = ConsoleEngine()",
                  f"eng.cmd_load_dataset({self.dataset_id!r}, "
                  f"n={0 if self.y is None else len(self.y)})",
                  f"active = {active!r}  # non-muted columns; eng.X is RAW original units "
                  "(hifi_anova quantile-transforms internally)",
                  "X = eng.X[:, active]",
                  "y = eng.y"]
        else:
            L += [f"# data: '{self.dataset_label}' ({0 if self.y is None else len(self.y)} rows) "
                  "— load YOUR data in ORIGINAL units (columns match `names` below):",
                  "# import pandas as pd",
                  "# df = pd.read_csv('your_file.csv')",
                  f"# X = df[{nms!r}].to_numpy()",
                  "# y = df['<your target column>'].to_numpy()",
                  "X = ...  # TODO: (N, D) feature matrix, columns in the `names` order below",
                  "y = ...  # TODO: (N,) target"]
        L += [f"names = {nms!r}",
              ""]
        if self.residual:
            L += ["# complement: fitted POST-HOC in the console (COMPLEMENT bus) at the"
                  " λ_res above;",
                  "# reproduced here as the pipeline's Stage C — same construction,"
                  " same split/seed."]
            if self.residual.get("hetero"):
                L += ["# NB Stage-D fit: the pipeline runs Stage C BEFORE Stage D,"
                      " but the console",
                      "# fitted this residual on the Stage-D-adapted mean's leftover"
                      " — expect the",
                      "# construction to reproduce, NOT byte-identical residual"
                      " weights."]
        L += ["res = ha.hifi_anova(",
              "    X, y, feature_names=names,",
              kw_lines,
              ")",
              "res.summary()"]
        return "\n".join(L)

    # ------------------------------------------------------- VERIFY LOO (Ses07)
    def cmd_verify_loo(self) -> Dict[str, Any]:
        """Run the Tier-III exact nested-refit LOO oracle for the CURRENT fit.

        ``exact_loo_nll`` refits the joint (mean + variance) estimator once per
        deleted observation — O(N) full refits, expensive — so it runs through
        the job queue like a fit (cancellable, progress events), never inline.
        It is the authority whenever the Tier-II guarantee is at risk (variance
        floor binding / ill-conditioned H_h — the R14 lamp's ✗ case).

        Honest scoping: on a homoscedastic fit (or when Stage-D's guard skipped
        the variance model) the three tiers coincide and the reported closed-form
        LOO is already exact — there is nothing to verify, and the command says
        so instead of burning O(N) refits."""
        with self._lock:
            res = self.result
        if res is None or not self.view:
            return {"ok": False, "error": "no fit to verify — FIT first"}
        if self.stale:
            return {"ok": False, "error": "mix is stale — refit, then VERIFY"}
        if self._fitting():
            return {"ok": False, "error": "fit running — verify after it finishes"}
        if self._active_job == "verify" or self._verify_pending:
            return {"ok": False, "error": "VERIFY LOO already running", "busy": True}
        rec = getattr(res, "_fitted_design", None)
        if rec is None:
            return {"ok": False, "error": "no fitted design on this result"}
        if not (getattr(rec, "is_weighted", False)
                and getattr(rec, "variance", None) is not None):
            return {"ok": False, "error":
                    "tiers coincide on this fit — the reported LOO is already "
                    "exact (closed-form leverage identity; homoscedastic or no "
                    "variance model). The Tier-III oracle verifies Stage-D fits."}
        self._cancel_warm()
        self._verify_pending = True
        self._jobs.put(("verify", {"epoch": self._epoch}))
        self._ensure_worker()
        return {"ok": True, "n_folds": int(rec.Phi.shape[0])}

    def _verify_worker(self, p: Dict[str, Any]) -> None:
        """Tier-III oracle job: chunked ``exact_loo_nll`` so it can report
        progress and honor cooperative cancel between chunks. Stores the result
        in ``self.verify`` (cleared on any stale change) and emits
        ``verify_done``; never touches the fit/view."""
        with self._lock:
            res = self.result
        rec = getattr(res, "_fitted_design", None) if res is not None else None
        if (res is None or rec is None or p.get("epoch") != self._epoch
                or self.stale):
            return
        from hifi_anova.analysis.automl import exact_loo_nll  # flagged import
        N = int(np.asarray(rec.Phi).shape[0])
        y = np.asarray(rec.y_centered, float) + float(rec.f0)
        t0 = time.time()
        self._emit({"event": "progress", "stage": "T3", "fraction": 0.0,
                    "message": f"TIER-III oracle — {N} exact joint refits"})
        per_point = np.empty(N)
        chunk = max(1, N // 40)
        done = 0
        while done < N:
            if self._stop.is_set():
                self._emit({"event": "warn",
                            "message": f"VERIFY LOO cancelled at {done}/{N}"})
                return
            idxs = np.arange(done, min(done + chunk, N))
            out = exact_loo_nll(rec.Phi, y, rec.reg_diag,
                                rec.variance.Psi, rec.variance.reg_var,
                                subset=idxs)
            per_point[idxs] = np.asarray(out["per_point_nll"], float)
            done = int(idxs[-1]) + 1
            self._emit({"event": "progress", "stage": "T3",
                        "fraction": _f(done / N),
                        "message": f"TIER-III oracle refit {done}/{N}"})
        loo3 = float(np.mean(per_point))
        m = (self.view or {}).get("master", {}) or {}
        reported = m.get("loo_nll")
        # authoritative dataset-row mapping (BR-07, DEC-055): the training rows
        # from ``res.split_indices['train']`` in Phi order (same convention as
        # view.diag — no seed reconstruction, no y-verify guessing).
        rows = None
        try:
            ti = np.asarray(res.split_indices["train"], int)
            if len(ti) == N:
                rows = ti
        except Exception:
            rows = None
        if rows is None:
            self._emit({"event": "warn", "message":
                        "train-row mapping unavailable — VERIFY worst rows "
                        "show train positions, not dataset ids"})
        worst = np.argsort(-per_point)[:8]
        with self._lock:
            self.verify = {
                "loo_nll_oracle": _f(loo3),
                "loo_nll_reported": reported,
                "delta": (None if reported is None
                          else _f(loo3 - float(reported))),
                "tier_reported": m.get("loo_tier"),
                "tier2_ok": m.get("tier2_ok"),
                "n_folds": N,
                "seconds": round(time.time() - t0, 2),
                # the rows the model predicts worst OUT-OF-SAMPLE (oracle NLL)
                "worst": [{"i": int(i),
                           "row": (int(rows[i]) if rows is not None else None),
                           "nll": _f(per_point[i])} for i in worst],
            }
        self._emit_done({"event": "verify_done",
                         "seconds": self.verify["seconds"]})

    # ------------------------------------------------- ΔLOO TEST (R25, rank-only)
    def cmd_loo_test(self, drops: bool = True, adds: bool = True) -> Dict[str, Any]:
        """Rank candidate structure changes by their PAIRED exact-LOO change.

        Rank-only, EXPLORATORY: for each candidate (mute an active variable /
        patch an unpatched pair) run a real refit and report the change in the
        exact plug-in LOO against the current fit — with a PAIRED per-point SE
        (base and candidate share the identical seeded train split, verified
        against y; per-point deleted-residual losses correlate ~0.95, so the
        paired SE is up to ~6x tighter than treating the two LOOs as
        independent). NO threshold, NO keep/drop rule, NO auto-apply — the
        stopping rule is expert-gated (X9C_gui3_loo_selection_brief.md, Q3').

        Honest scoping (refusals, not workarounds):
        - Homoscedastic fits only: the paired route needs the unweighted
          closed-form LOO (``ridge_analytics``, ``sample_weights=None``) — a
          hetero ΔLOO-NLL inherits the E8 hetero-path question and is NOT
          approximated here.
        - Pair-add candidates are skipped on a mixed per-channel-K fit: the
          backend forces uniform K for pair fits, so the Δ would conflate the
          pair with a K change and could not be attributed to the pair.
        - A drop that also removes patched pairs, and clique-closure extras
          induced by an add, are annotated on the row — the Δ is attributed to
          the ACTUAL model change, never to the clicked candidate alone.
        """
        with self._lock:
            res = self.result
        if res is None or not self.view:
            return {"ok": False, "error": "no fit to test against — FIT first"}
        if self.stale:
            return {"ok": False, "error": "mix is stale — refit, then ΔLOO TEST"}
        if self._fitting():
            return {"ok": False, "error": "fit running — test after it finishes"}
        if self._active_job == "lootest" or self._loo_test_pending:
            return {"ok": False, "error": "ΔLOO TEST already running", "busy": True}
        rec = getattr(res, "_fitted_design", None)
        if rec is None:
            return {"ok": False, "error": "no fitted design on this result"}
        if getattr(rec, "sample_weights", None) is not None:
            return {"ok": False, "error":
                    "ΔLOO TEST is homoscedastic-only: the paired exact-LOO "
                    "route needs the unweighted closed-form identity; a "
                    "GLS/Stage-D ΔLOO-NLL is the hetero-path question (E8) "
                    "and is not approximated here"}
        if not (drops or adds):
            return {"ok": False, "error": "nothing to test (drops and adds both off)"}
        self._cancel_warm()
        self._loo_test_pending = True
        self._jobs.put(("lootest", {"epoch": self._epoch,
                                    "drops": bool(drops), "adds": bool(adds)}))
        self._ensure_worker()
        return {"ok": True}

    def _loo_test_candidates(self, nms: List[str], ks: List[int],
                             drops: bool, adds: bool):
        """Enumerate (candidates, notes). Bounded fan-out; caps are REPORTED."""
        notes: List[str] = []
        cands: List[Dict[str, Any]] = []
        pos = {nm: i for i, nm in enumerate(nms)}
        cur_pairs, cur_endpoints = self._active_pairs(nms)
        if drops:
            if len(nms) < 2:
                notes.append("drops skipped: only one active channel")
            else:
                dlist = list(nms)
                if len(dlist) > LOO_TEST_MAX_DROPS:
                    notes.append(f"drops capped at {LOO_TEST_MAX_DROPS} of "
                                 f"{len(dlist)} active channels")
                    dlist = dlist[:LOO_TEST_MAX_DROPS]
                for nm in dlist:
                    removed = [p for p in cur_pairs if nm in p]
                    cands.append({"kind": "drop", "label": f"− {nm}", "name": nm,
                                  "note": ("also removes pair "
                                           + ", ".join("×".join(p) for p in removed)
                                           if removed else None)})
        if adds:
            if len(set(ks)) > 1:
                notes.append("pair-adds skipped: mixed per-channel K — the "
                             "backend forces uniform K for pair fits, so a Δ "
                             "would conflate the pair with a K change")
            elif len(nms) < 2:
                pass
            else:
                cur_set = {frozenset(p) for p in cur_pairs}
                # base admitted set is the clique over the current endpoints
                base_adm = {frozenset((nms[i], nms[j]))
                            for ai, i in enumerate(cur_endpoints)
                            for j in cur_endpoints[ai + 1:]}
                todo = [(a, b) for i, a in enumerate(nms) for b in nms[i + 1:]
                        if frozenset((a, b)) not in cur_set
                        and frozenset((a, b)) not in base_adm]
                # order by SCAN residual-capture score when a fresh scan exists
                sc = (self.scan or {}).get("scores") or {}
                todo.sort(key=lambda p: -(sc.get(f"{p[0]}|{p[1]}")
                                          or sc.get(f"{p[1]}|{p[0]}") or 0.0))
                if len(todo) > LOO_TEST_MAX_ADDS:
                    notes.append(f"pair-adds capped at {LOO_TEST_MAX_ADDS} of "
                                 f"{len(todo)} unpatched pairs"
                                 + (" (SCAN-ranked)" if sc else
                                    " (unranked — SCAN first to rank the cap)"))
                    todo = todo[:LOO_TEST_MAX_ADDS]
                for a, b in todo:
                    ep = sorted(set(cur_endpoints) | {pos[a], pos[b]})
                    adm = {frozenset((nms[i], nms[j]))
                           for ai, i in enumerate(ep) for j in ep[ai + 1:]}
                    induced = adm - base_adm - {frozenset((a, b))}
                    cands.append({"kind": "add", "label": f"+ {a}×{b}",
                                  "a": a, "b": b, "endpoints": ep,
                                  "note": ("clique closure also admits "
                                           + ", ".join("×".join(sorted(p))
                                                       for p in sorted(induced, key=sorted))
                                           if induced else None)})
        return cands, notes

    def _loo_test_worker(self, p: Dict[str, Any]) -> None:
        """ΔLOO TEST job: one real refit per candidate through this worker
        (cancellable between AND during fits — a user fit preempts), then the
        paired per-point deleted-residual delta vs the base fit. Stores the
        ranked list in ``self.loo_test`` (cleared on any stale change); never
        touches the fit/view."""
        import hifi_anova as ha
        from hifi_anova.analysis.automl import ridge_analytics  # flagged import
        with self._lock:
            res = self.result
        rec = getattr(res, "_fitted_design", None) if res is not None else None
        if (res is None or rec is None or p.get("epoch") != self._epoch
                or self.stale):
            return
        t0 = time.time()
        active = [i for i, nm in enumerate(self.names) if nm not in self.muted]
        nms = [self.names[i] for i in active]
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in nms]
        cands, notes = self._loo_test_candidates(nms, ks, p["drops"], p["adds"])
        if not cands:
            with self._lock:
                self.loo_test = {"rows": [], "notes": notes, "n_candidates": 0,
                                 "base_loo_r2": None, "seconds": 0.0,
                                 "exploratory": True}
            self._emit_done({"event": "loo_test_done", "seconds": 0.0})
            return
        # base: exact plug-in LOO per point, same instrument as the master meter
        an0 = ridge_analytics(np.asarray(rec.Phi, float),
                              np.asarray(rec.y_centered, float),
                              np.asarray(rec.reg_diag, float))
        r0sq = np.asarray(an0["loo_residuals"], float) ** 2
        y0 = np.asarray(rec.y_centered, float) + float(rec.f0)
        df0 = float(an0["df"])
        var_y = float(np.var(self.y))
        base_r2 = (1.0 - float(an0["loo_cv"]) / var_y) if var_y > 0 else None
        K = len(cands)
        self._emit({"event": "progress", "stage": "ΔLOO", "fraction": 0.0,
                    "message": f"ΔLOO TEST — {K} candidate refits"})
        rows: List[Dict[str, Any]] = []
        for ci, c in enumerate(cands):
            if self._stop.is_set():
                self._emit({"event": "warn",
                            "message": f"ΔLOO TEST cancelled at {ci}/{K}"})
                return
            try:
                if c["kind"] == "drop":
                    keep = [j for j, nm in enumerate(nms) if nm != c["name"]]
                    nms2 = [nms[j] for j in keep]
                    ks2 = [ks[j] for j in keep]
                    if self._active_pairs(nms)[1]:
                        # the base pair fit forced uniform K1=max(ks) on every
                        # channel — survivors must KEEP that K, else dropping
                        # the max-K variable would also lower everyone's
                        # fidelity and the Δ would conflate the two changes
                        ks2 = [max(ks)] * len(nms2)
                    xs2 = self.X[:, [active[j] for j in keep]]
                    _, ep2 = self._active_pairs(nms2)
                    kwargs = self._fit_kwargs(ks2, nms2, hetero=False,
                                              pairs=ep2 or None)
                else:
                    nms2, ks2 = nms, ks
                    xs2 = self.X[:, active]
                    kwargs = self._fit_kwargs(ks, nms, hetero=False,
                                              pairs=c["endpoints"])
                res2 = ha.hifi_anova(xs2, self.y, feature_names=nms2, **kwargs)
                rec2 = res2._fitted_design
                if getattr(rec2, "sample_weights", None) is not None:
                    rows.append({**{k: c[k] for k in ("kind", "label", "note")},
                                 "error": "candidate fit came back GLS-weighted"
                                          " — paired homoscedastic ΔLOO "
                                          "unavailable"})
                    continue
                y1 = np.asarray(rec2.y_centered, float) + float(rec2.f0)
                tol = 1e-8 * max(1.0, float(np.max(np.abs(y0))))
                if y1.shape != y0.shape or not np.allclose(y0, y1, atol=tol):
                    # rows not verifiably the same observations → no honest
                    # pairing exists; say so instead of quoting an unpaired SE
                    rows.append({**{k: c[k] for k in ("kind", "label", "note")},
                                 "error": "train rows differ from the base "
                                          "fit — paired ΔLOO unavailable"})
                    continue
                an1 = ridge_analytics(np.asarray(rec2.Phi, float),
                                      np.asarray(rec2.y_centered, float),
                                      np.asarray(rec2.reg_diag, float))
                r1sq = np.asarray(an1["loo_residuals"], float) ** 2
                d = r1sq - r0sq          # per-point LOO loss change (paired)
                n = int(d.size)
                se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else None
                rows.append({
                    **{k: c[k] for k in ("kind", "label", "note")},
                    # ΔLOO-R² > 0 ⇒ the candidate predicts better out-of-sample
                    "delta_loo_r2": _f(-float(np.mean(d)) / var_y) if var_y > 0 else None,
                    "se_loo_r2": _f(se / var_y) if (se is not None and var_y > 0) else None,
                    "delta_df": _f(float(an1["df"]) - df0),
                    "n": n,
                })
            except ha.HiFiCancelled:
                self._emit({"event": "warn",
                            "message": f"ΔLOO TEST cancelled at {ci}/{K}"})
                return
            except Exception as exc:
                # a failed candidate is reported, never silently dropped
                rows.append({**{k: c[k] for k in ("kind", "label", "note")},
                             "error": f"{type(exc).__name__}: {exc}"})
            self._emit({"event": "progress", "stage": "ΔLOO",
                        "fraction": _f((ci + 1) / K),
                        "message": f"ΔLOO TEST {ci + 1}/{K} · {c['label']}"})
        # ranked: best predictive gain first; failed rows sink to the bottom
        rows.sort(key=lambda r: (r.get("delta_loo_r2") is None,
                                 -(r.get("delta_loo_r2") or 0.0)))
        with self._lock:
            if p.get("epoch") != self._epoch or self.stale:
                self._emit({"event": "warn", "message":
                            "ΔLOO TEST finished but the mix changed — "
                            "results discarded (they rank against a stale fit)"})
                return
            self.loo_test = {
                "rows": rows, "notes": notes, "n_candidates": K,
                "base_loo_r2": _f(base_r2),
                "seconds": round(time.time() - t0, 2),
                "exploratory": True,   # same-data selection — never confirmatory
            }
        self._emit_done({"event": "loo_test_done",
                         "seconds": self.loo_test["seconds"]})

    # ------------------------------------------- RESIDUAL bus (C16, X12C Phase 1)
    def cmd_fit_residual(self, family: Optional[str] = None, n_centers=None,
                         width=None, lam=None,
                         kernel: Optional[str] = None) -> Dict[str, Any]:
        """Fit a smooth catch-all residual on the CURRENT fit's leftover.

        A post-hoc, decoupled second solve — NEVER reruns Stages A/B: the
        residual features are projected exactly orthogonal to the fitted
        Fourier design (in-sample), so the main fit's coefficients and the
        Sobol attribution are untouched by construction (the shipped Stage-C
        guarantee, ``core/projection.py``). EXPLORATORY by design; the
        population Sobol decomposition may still shift (in-sample caveat).

        ``family``: nystrom (default — GP inducing-point approximation) / rbf
        / rff. ``n_centers`` maps to the family's own feature count (library
        default when omitted). ``width``: explicit override; omitted → the
        engine PROPOSES the criterion-min width over a DISCLOSED compute grid
        (Phase 1.5 width profile, ~9 log-spaced multiples of the library
        default; cached per fit) and uses it — falling back to the library
        default when the profile is unavailable (M above the profile cap).
        ``lam``: explicit λ_res override; omitted → the engine PROPOSES the
        exact-LOO minimum along a closed-form λ path (``RidgePathEigSolver``)
        and uses it. Both proposals are disclosed first guesses, the fader
        override always wins (R16 class; default blessing is expert-gated,
        analysis §6.2/§6.4).

        Runs as job kind ``"residual"`` on the single worker (a user FIT
        preempts it). Honest refusals: no fit / stale / fit running /
        duplicate in-flight. Design-layout refusals are GONE (BR-11, Ses06):
        the projector stores and rebuilds the SOLVED layout (mixed
        per-channel K, per-pair K₂, order-selective membership, and the
        intercept-only complement-only limit) via the shared
        ``core.features.build_mean_design`` builder."""
        with self._lock:
            res = self.result
        if res is None or not self.view:
            return {"ok": False, "error": "no fit to extend — FIT first"}
        if self.stale:
            return {"ok": False, "error": "mix is stale — refit, then FIT COMPLEMENT"}
        if self._fitting():
            return {"ok": False, "error": "fit running — fit the complement after it"}
        if self._active_job == "residual" or self._residual_pending:
            return {"ok": False, "error": "residual fit already running", "busy": True}
        family = str(family or self.residual_cfg.get("family")
                     or "nystrom").lower()
        if family not in RESIDUAL_FAMILIES:
            return {"ok": False, "error": f"unknown residual family {family} "
                                          f"({'/'.join(RESIDUAL_FAMILIES)})"}
        if n_centers is not None:
            n_centers = int(n_centers)
            if not (2 <= n_centers <= RESIDUAL_M_MAX):
                return {"ok": False,
                        "error": f"n_centers must be 2..{RESIDUAL_M_MAX}"}
        if width is not None:
            width = float(width)
            if not (width > 0):
                return {"ok": False, "error": "width must be > 0"}
        # NYSTRÖM Matérn ν stepper (§4): a user CHOICE like the family select
        # (library default rbf) — refused honestly on the other families
        # rather than silently ignored.
        if kernel is not None:
            kernel = str(kernel).lower()
            if kernel not in RES_KERNELS:
                return {"ok": False, "error": f"unknown nystrom kernel "
                        f"{kernel} ({'/'.join(RES_KERNELS)})"}
            if family != "nystrom":
                return {"ok": False, "error": "kernel selection applies to "
                        "the NYSTRÖM family only"}
        if lam is not None:
            lam = float(lam)
            if not (RES_LAMBDA_MIN <= lam <= RES_LAMBDA_MAX):
                return {"ok": False, "error": "lambda_res out of range"}
        # BR-11 (Ses06): the residual projector now stores and rebuilds the
        # SOLVED design layout (mixed per-channel K, per-pair K₂ pinning,
        # order-selective membership — core.features.build_mean_design shared
        # with the model), so the historical uniform-layout refusals are gone.
        # The worker still verifies the orthogonality premise per width against
        # rec.Phi and degrades openly if it ever fails.
        self.residual_cfg = {"family": family, "n_centers": n_centers,
                             "width": width, "lam": lam, "kernel": kernel}
        self._cancel_warm()
        self._residual_pending = True
        self._jobs.put(("residual", {"epoch": self._epoch, "family": family,
                                     "n_centers": n_centers, "width": width,
                                     "lam": lam, "kernel": kernel}))
        self._ensure_worker()
        return {"ok": True, "family": family}

    def cmd_clear_residual(self) -> Dict[str, Any]:
        """Detach the residual (✕): the model returns to the pure structured
        fit; also cancels an in-flight residual job. The view is rebuilt for
        the base model so PARITY/probe drop back honestly."""
        dropped = self._drain_jobs({"residual"})
        if self._active_job == "residual":
            self._stop.set()
        self._residual_pending = False
        if "residual" in dropped:   # queued job dropped — worker never saw it
            self._emit({"event": "residual_discarded", "reason": "detached"})
        with self._lock:
            res = self.result
            had = self._residual_base_model is not None
            if res is not None and had:
                res.model = self._residual_base_model
            self._residual_base_model = None
            self.residual = None
            active = list(self._last_active or [])
            nms = list(self._active_names)
        if had and res is not None and active and not self.stale \
                and not self._fitting():
            try:
                old_path = (self.view or {}).get("path")
                view, yhat, xt = self._build_view(res, active, nms, warn=False)
                view["path"] = old_path
                with self._lock:
                    self.view, self._yhat, self._xt = view, yhat, xt
            except Exception as exc:
                self._emit({"event": "warn", "message":
                            "view rebuild after residual clear failed: "
                            f"{type(exc).__name__}: {exc}"})
        self._emit({"event": "residual_cleared"})
        return {"ok": True, "cleared": bool(had)}

    def _detach_residual(self) -> None:
        """Drop an attached residual (model back to base) WITHOUT rebuilding
        the view — used on stale-marking, where the view is outdated anyway
        and the next fit replaces it wholesale."""
        with self._lock:
            if self.result is not None and self._residual_base_model is not None:
                self.result.model = self._residual_base_model
            self._residual_base_model = None
            self.residual = None

    def _residual_worker(self, p: Dict[str, Any]) -> None:
        """COMPLEMENT job: the shipped Stage-C construction on the CURRENT fit.

        leftover + base instrument → WIDTH profile (Phase 1.5: per grid width,
        rebuild Z → project → one M×M eigh → λ path; criterion at the
        per-width-optimal λ; cached per fit) → applied width (user override,
        else the profile's proposal, else the library default) → build
        features → project orthogonal to the fitted Fourier design →
        closed-form λ_res criterion path at the applied width → selected λ
        (user override wins) → ridge solve → ``eqx.tree_at`` attach to a COPY
        of the model → rebuild the view with the combined model → total-order
        Sobol readout (§5c, never blocks the attach) → store the
        ``self.residual`` block.

        Criterion (disclosed, never train-R²): on a homoscedastic fit the
        curve is the COMBINED model's exact plug-in LOO — combined leverages
        are the SUM of the base and residual-block leverages (orthogonal
        blocks + block-diagonal penalty ⇒ the joint hat matrix is block-
        additive; verified numerically in tests, expert question §6.2) — so
        the block reports an honest ΔLOO-R² vs the base fit with a paired
        per-point SE (the R25 convention). On a GLS/Stage-D fit the base
        leverages live in the weighted instrument, so the curve degrades to
        the residual system's own exact LOO on the mean fit's leftover
        (disclosed; the residual solve itself is unweighted — efficiency loss
        only, analysis §1.3)."""
        import equinox as eqx
        from hifi_anova.analysis.automl import ridge_analytics  # flagged import
        from hifi_anova.training.hyperopt import RidgePathEigSolver  # flagged
        from hifi_anova.training.analytic_residual import (  # flagged import
            create_residual, _create_fitted_residual)
        from hifi_anova.core.projection import project_features_orthogonal
        from hifi_anova.model.linear_residual import predict_residual_batch
        with self._lock:
            res = self.result
            base_model = (self._residual_base_model
                          if self._residual_base_model is not None
                          else (res.model if res is not None else None))
        rec = getattr(res, "_fitted_design", None) if res is not None else None
        # EVERY worker exit without residual_done emits a residual_discarded
        # TERMINAL event (Ses07 follow-up to the race fix): a ws waiter
        # gating on the bus gets a settled-flag snapshot instead of needing a
        # timeout for the silent paths (via _emit_done — flushed after
        # _job_loop settles the busy flags).
        if (res is None or rec is None or base_model is None
                or p.get("epoch") != self._epoch or self.stale):
            self._emit_done({"event": "residual_discarded",
                             "reason": "fit changed"})
            return
        t0 = time.time()
        family = p["family"]
        m_key, w_key, w_default = RESIDUAL_PARAM_KEYS[family]
        cfg: Dict[str, Any] = {"seed": 42}
        if p.get("n_centers") is not None:
            cfg[m_key] = int(p["n_centers"])
        if p.get("width") is not None:
            cfg[w_key] = float(p["width"])
        if p.get("kernel"):                    # nystrom-only (validated at cmd)
            cfg["kernel"] = str(p["kernel"])
        # train rows in Phi order (BR-07/DEC-055 authoritative mapping)
        try:
            ti = np.asarray(res.split_indices["train"], int)
        except Exception:
            self._emit({"event": "warn", "message":
                        "COMPLEMENT unavailable: no train-row mapping on this fit"})
            self._emit_done({"event": "residual_discarded",
                             "reason": "no train-row mapping"})
            return
        x_tr = np.asarray(self._xt, float)[ti]
        y_tr = np.asarray(self.y, float)[ti]
        n_tr = len(y_tr)
        ytol = 1e-8 * max(1.0, float(np.max(np.abs(y_tr))))
        if not np.allclose(np.asarray(rec.y_centered, float) + float(rec.f0),
                           y_tr, atol=ytol):
            self._emit({"event": "warn", "message":
                        "COMPLEMENT unavailable: fitted design rows do not match "
                        "the train split (instrument mismatch)"})
            self._emit_done({"event": "residual_discarded",
                             "reason": "instrument mismatch"})
            return
        self._emit({"event": "progress", "stage": "RES", "fraction": 0.05,
                    "message": "COMPLEMENT — leftover + base instrument"})
        # 1) leftover of the CURRENT mean model (GLS-adapted coefficients
        #    included on a Stage-D fit; the residual solve itself is unweighted)
        phi1 = base_model.build_phi1(x_tr)
        phi2 = base_model.build_phi2(x_tr)
        phi3 = base_model.build_phi3(x_tr)
        mean_tr = np.asarray(
            base_model.mean_model.predict(phi1, phi2, phi3), float).reshape(-1)
        r = y_tr - mean_tr
        # 2) fitted-design Φ (the projection target) + the base-instrument
        #    premise (homoscedastic only): the same exact plug-in LOO the
        #    master meter / ΔLOO TEST use. The base-side premises (unweighted
        #    instrument; its residuals == the model's leftover) are width-
        #    independent, so they are checked ONCE; the orthogonality premise
        #    is per-width and re-checked on every projected block. Degrade in
        #    the open rather than quoting a combined number that isn't one.
        # solved-design layout (BR-11): the projection targets the columns the
        # base was actually solved on (== rec.Phi layout); on a BR-06 subset
        # fit the excluded first-order structure stays available to the
        # complement, and an intercept-only base gives (N, 0) ⇒ projection
        # no-op (complement-only)
        Phi_tr = base_model.build_phi_all_fit(x_tr)
        Phi_solved = np.asarray(rec.Phi, float)
        var_y = float(np.var(self.y))
        var_r = float(np.var(r))
        notes: List[str] = []
        combined_base = getattr(rec, "sample_weights", None) is None
        an0 = None
        if not combined_base:
            notes.append("GLS/Stage-D fit: combined exact LOO is unavailable "
                         "(weighted base instrument) — the curve is the "
                         "residual system's own exact LOO on the leftover; "
                         "the residual solve is unweighted (disclosed)")
        else:
            an0 = ridge_analytics(Phi_solved,
                                  np.asarray(rec.y_centered, float),
                                  np.asarray(rec.reg_diag, float))
            e0 = np.asarray(an0["residuals"], float)
            if not np.allclose(e0, r, atol=100 * ytol):
                combined_base = False
                an0 = None
                notes.append("base-instrument residuals differ from the "
                             "model's leftover — combined LOO degraded to "
                             "the residual-system curve")
        lev0 = (np.asarray(an0["leverages"], float) if an0 is not None
                else None)
        e_base = (np.asarray(an0["residuals"], float) if an0 is not None
                  else None)
        base_loo0 = float(an0["loo_cv"]) if an0 is not None else None
        orth_tol = 1e-6 * n_tr
        base_grid = np.logspace(RES_PATH_LOG10_MIN, RES_PATH_LOG10_MAX,
                                RES_PATH_POINTS)

        def _block_path(Zp_w, grid_w):
            """Shared λ-path machinery on one width's projected block: the
            per-width orthogonality premise + one M×M eigh, then every λ in
            O(N·M) (flagged private access: path leverages need the
            eigenbasis)."""
            orth_w = (float(np.max(np.abs(Phi_solved.T @ Zp_w)))
                      if n_tr and Phi_solved.size else 0.0)
            comb_w = combined_base and orth_w <= orth_tol
            solver = RidgePathEigSolver(Zp_w, r, np.ones(Zp_w.shape[1]))
            mu = solver._mu
            T = solver._Phi @ solver._Q
            T2 = T * T
            G = len(grid_w)
            loo_mse = np.empty(G)
            gcv = np.empty(G)
            dfs = np.empty(G)
            for i, lam in enumerate(grid_w):
                dg = solver.diagnostics(lam)
                zfit = Zp_w @ dg["w"]
                hZ = T2 @ (1.0 / (mu + lam))
                if comb_w:
                    h = np.clip(lev0 + hZ, 0.0, 1.0 - 1e-10)
                    e = e_base - zfit
                else:
                    h = np.clip(hZ, 0.0, 1.0 - 1e-10)
                    e = r - zfit
                loo_mse[i] = float(np.mean((e / (1.0 - h)) ** 2))
                gcv[i] = dg["gcv"]
                dfs[i] = dg["df"]
            return {"solver": solver, "mu": mu, "T2": T2,
                    "loo_mse": loo_mse, "gcv": gcv, "dfs": dfs,
                    "combined": comb_w, "orth_max": orth_w}

        # 3) WIDTH profile (Phase 1.5, analysis §4): criterion vs width at
        #    the per-width-optimal λ over a DISCLOSED compute grid (grid
        #    blessing is §6.4 brief material), cached per (fit, family, M);
        #    skipped above the M cap (one M×M eigh per grid point).
        m_eff = min((int(p["n_centers"]) if p.get("n_centers") is not None
                     else RESIDUAL_M_DEFAULT[family]), n_tr)
        prof_key = (p.get("epoch"), family, p.get("n_centers"),
                    p.get("kernel"))       # a kernel change reshapes Z too
        profile = None
        cache = self._res_width_profile
        if cache is not None and cache.get("key") == prof_key:
            profile = dict(cache["block"])
            profile["cached"] = True
        elif m_eff > RES_WIDTH_PROFILE_M_MAX:
            notes.append(f"width profile skipped (M={m_eff} > "
                         f"{RES_WIDTH_PROFILE_M_MAX}: one M×M eigh per grid "
                         "point) — AUTO width falls back to the library "
                         "default")
        else:
            w_grid = [w_default * f for f in RES_WIDTH_GRID_FACTORS]
            vals: List[Optional[float]] = []
            lam_opts: List[float] = []
            for j, w in enumerate(w_grid):
                if self._stop.is_set():
                    self._emit_done({"event": "residual_discarded",
                                     "reason": "cancelled"})
                    return
                self._emit({"event": "progress", "stage": "RES",
                            "fraction": 0.08 + 0.27 * j / len(w_grid),
                            "message": f"COMPLEMENT — width grid "
                                       f"{j + 1}/{len(w_grid)}"})
                cfg_w = dict(cfg)
                cfg_w[w_key] = float(w)
                obj_w = create_residual(family, cfg_w, x_tr, x_tr.shape[1],
                                        key=None)
                Zpw = np.asarray(project_features_orthogonal(
                    obj_w.build_features(x_tr), Phi_tr)[0], float)
                st = _block_path(Zpw, base_grid)
                i_opt = int(np.argmin(st["loo_mse"]))
                lam_opts.append(float(base_grid[i_opt]))
                best = float(st["loo_mse"][i_opt])
                if combined_base and not st["combined"]:
                    vals.append(None)   # this width's projection degraded
                elif combined_base:
                    vals.append((base_loo0 - best) / var_y
                                if var_y > 0 else 0.0)
                else:
                    vals.append(1.0 - best / var_r if var_r > 0 else 0.0)
            if any(v is not None for v in vals):
                iw = int(max((k for k in range(len(vals))
                              if vals[k] is not None),
                             key=lambda k: vals[k]))
                profile = {
                    "widths": [_f(v) for v in w_grid],
                    "log10_width": [_f(np.log10(v)) for v in w_grid],
                    "curve": [_f(v) for v in vals],
                    "curve_kind": ("delta_loo_r2" if combined_base
                                   else "leftover_loo_r2"),
                    "lambda_opt": [_f(v) for v in lam_opts],
                    "i_min": iw, "cached": False,
                    "grid_note": (f"×{RES_WIDTH_GRID_FACTORS[0]:.2g}–"
                                  f"×{RES_WIDTH_GRID_FACTORS[-1]:.2g} around "
                                  f"the library default {w_default:g} "
                                  f"({len(w_grid)} log-spaced points) — "
                                  "disclosed compute grid, not reviewed "
                                  "statistics"),
                }
                self._res_width_profile = {"key": prof_key, "block": profile}
                profile = dict(profile)
            else:
                notes.append("width profile unavailable: every grid width "
                             "lost the projection premise — AUTO width "
                             "falls back to the library default")
        # width applied: fader override wins; AUTO = the profile's proposed
        # (criterion-min) width when a profile exists, else the library
        # default — R16-class disclosed first guess either way (§6.4).
        w_user = p.get("width")
        w_prop = (float(profile["widths"][profile["i_min"]])
                  if profile is not None else None)
        if w_user is not None:
            width_source = "user"
            cfg[w_key] = float(w_user)
        elif w_prop is not None:
            width_source = "auto"
            cfg[w_key] = w_prop
        else:
            width_source = "auto"      # library default (profile absent)
        if self._stop.is_set():
            self._emit_done({"event": "residual_discarded",
                             "reason": "cancelled"})
            return
        self._emit({"event": "progress", "stage": "RES", "fraction": 0.38,
                    "message": f"COMPLEMENT — {family} features"})
        # 4) features + exact orthogonal projection at the APPLIED width
        residual_obj = create_residual(family, cfg, x_tr, x_tr.shape[1],
                                       key=None)
        Z = residual_obj.build_features(x_tr)
        if self._stop.is_set():
            self._emit_done({"event": "residual_discarded",
                             "reason": "cancelled"})
            return
        Z_proj, proj_coeffs = project_features_orthogonal(Z, Phi_tr)
        Zp = np.asarray(Z_proj, float)
        M = Zp.shape[1]
        self._emit({"event": "progress", "stage": "RES", "fraction": 0.5,
                    "message": f"COMPLEMENT — λ path (M={M})"})
        # 5) closed-form λ_res path at the applied width (one M×M eigh)
        lam_user = p.get("lam")
        grid = base_grid
        if lam_user is not None and (RES_PATH_LOG10_MIN
                                     <= np.log10(lam_user)
                                     <= RES_PATH_LOG10_MAX):
            grid = np.unique(np.append(grid, float(lam_user)))
        pstats = _block_path(Zp, grid)
        solver, mu, T2 = pstats["solver"], pstats["mu"], pstats["T2"]
        loo_mse, gcv, dfs = pstats["loo_mse"], pstats["gcv"], pstats["dfs"]
        combined = pstats["combined"]
        orth_max = pstats["orth_max"]
        if combined_base and not combined:
            notes.append(f"projection not orthogonal to the solved design "
                         f"(max |ΦᵀZ|={orth_max:.2e}) — combined LOO "
                         "degraded to the residual-system curve")
        i_min = int(np.argmin(loo_mse))
        i_gcv = int(np.argmin(gcv))
        lam_prop = float(grid[i_min])
        lam_used = float(lam_user) if lam_user is not None else lam_prop
        # 6) solve at the applied λ, attach to a COPY of the model
        dg = solver.diagnostics(lam_used)
        alpha = dg["w"]
        fitted = _create_fitted_residual(residual_obj, alpha,
                                         proj_coeffs, base_model)
        combined_model = eqx.tree_at(lambda m: m.residual_net, base_model,
                                     fitted, is_leaf=lambda x: x is None)
        zfit = Zp @ alpha
        hZ_used = T2 @ (1.0 / (mu + lam_used))
        df_z = float(np.sum(mu / (mu + lam_used)))
        tr_hz2 = float(np.sum((mu / (mu + lam_used)) ** 2))
        delta = se = comb_r2 = base_r2 = df_comb = sigma2_comb = None
        guard = None
        curve_kind = "delta_loo_r2"
        if combined:
            base_loo = float(an0["loo_cv"])
            base_r2 = (1.0 - base_loo / var_y) if var_y > 0 else None
            h_c = np.clip(lev0 + hZ_used, 0.0, 1.0 - 1e-10)
            e_c = e_base - zfit
            loo_c = e_c / (1.0 - h_c)
            mse_c = float(np.mean(loo_c ** 2))
            if var_y > 0:
                delta = _f((base_loo - mse_c) / var_y)
                comb_r2 = _f(1.0 - mse_c / var_y)
                # paired per-point SE — the R25 ΔLOO TEST convention (base and
                # combined share the identical rows by construction)
                d = loo_c ** 2 - np.asarray(an0["loo_residuals"], float) ** 2
                se = _f(float(np.std(d, ddof=1) / np.sqrt(n_tr)) / var_y) \
                    if n_tr > 1 else None
            df_comb = _f(float(an0["df"]) + df_z)
            # combined implied noise with COMBINED residual df — tr(H_Φ·H_Z)=0
            # for orthogonal blocks, so tr(H²) is additive too (§1.3 leverage
            # lesson: never quote a post-residual σ̂ off base bookkeeping)
            rss_c = float(np.sum(e_c ** 2))
            df_res_c = max(n_tr - 2.0 * (float(an0["df"]) + df_z)
                           + float(an0["tr_H2"]) + tr_hz2, 1.0)
            sigma2_comb = _f(rss_c / df_res_c)
            mf = self.noise_model_free or {}
            if sigma2_comb is not None and mf.get("sigma2") is not None:
                guard = {"floor_sigma2": _f(mf["sigma2"]),
                         "method": mf.get("method"),
                         "triggered": bool(sigma2_comb < float(mf["sigma2"]))}
            curve = ((base_loo - loo_mse) / var_y if var_y > 0
                     else np.zeros_like(loo_mse))
        else:
            curve = (1.0 - loo_mse / var_r if var_r > 0
                     else np.zeros_like(loo_mse))
            curve_kind = "leftover_loo_r2"
        if family == "rff":
            # RFFResidual bakes γ into its frequency draws (no attribute
            # survives ``create``) — report the APPLIED value: the override/
            # proposal in cfg, else the library default. (Ses03 fix: the old
            # ``getattr(obj, 'gamma')`` crashed the whole worker on RFF.)
            width_eff = float(cfg.get(w_key, w_default))
        else:
            width_eff = float(getattr(residual_obj, {
                "rbf": "sigma", "nystrom": "lengthscale"}[family]))
        if self._stop.is_set():
            self._emit_done({"event": "residual_discarded",
                             "reason": "cancelled"})
            return
        self._emit({"event": "progress", "stage": "RES", "fraction": 0.8,
                    "message": "COMPLEMENT — attach + view rebuild"})
        with self._lock:
            # _stop re-checked HERE: a ✕/cancel that lands after the last
            # cooperative checkpoint must still win — never attach past it
            if (p.get("epoch") != self._epoch or self.stale
                    or self.result is not res or self._stop.is_set()):
                self._emit_done({"event": "residual_discarded",
                                 "reason": ("cancelled" if self._stop.is_set()
                                            else "fit changed")})
                return
            res.model = combined_model
            self._residual_base_model = base_model
            active = list(self._last_active or [])
            nms = list(self._active_names)
        # captured share over ALL rows (ledger scale), in-sample by nature
        g_all = np.asarray(predict_residual_batch(fitted, self._xt),
                           float).reshape(-1)
        captured = _f(float(np.var(g_all)) / var_y) if var_y > 0 else None
        try:
            old_path = (self.view or {}).get("path")
            view, yhat, xt = self._build_view(res, active, nms, warn=False)
            view["path"] = old_path
        except Exception as exc:
            self._detach_residual()
            self._emit({"event": "warn", "message":
                        "COMPLEMENT view rebuild failed — detached: "
                        f"{type(exc).__name__}: {exc}"})
            self._emit_done({"event": "residual_discarded",
                             "reason": "view rebuild failed"})
            return
        # 7) total-order Sobol "rides on" readout (Phase 1.5, §5c) — never
        #    blocks the attach: a failure degrades to a note, not an error
        self._emit({"event": "progress", "stage": "RES", "fraction": 0.9,
                    "message": "COMPLEMENT — Sobol readout (QMC)"})
        sobol_block = None
        try:
            sobol_block = self._residual_sobol_readout(fitted, base_model,
                                                       nms)
        except Exception as exc:
            notes.append("Sobol readout unavailable: "
                         f"{type(exc).__name__}: {exc}")
        block = {
            "family": family, "m": int(M), "width": _f(width_eff),
            "kernel": (str(getattr(residual_obj, "kernel_type", ""))
                       if family == "nystrom" else None),
            "lambda": _f(lam_used),
            "lambda_log10": _f(np.log10(lam_used)),
            "lambda_source": "user" if lam_user is not None else "auto",
            "lambda_proposed": _f(lam_prop),
            # WIDTH profile (Phase 1.5): AUTO width = the profile's proposed
            # criterion-min width over the DISCLOSED grid (library default
            # when the profile is unavailable); the fader override wins.
            "width_source": width_source,
            "width_proposed": _f(w_prop),
            "width_profile": profile,
            "sobol": sobol_block,
            # the DISCLOSED selection criterion (R16 class — a labeled first
            # guess, not reviewed statistics; default blessing is §6.2)
            "criterion": ("combined exact-LOO minimum"
                          if curve_kind == "delta_loo_r2"
                          else "residual-system exact-LOO minimum (leftover)"),
            "path": {"log10_lambda": [_f(v) for v in np.log10(grid)],
                     "curve": [_f(v) for v in curve],
                     "curve_kind": curve_kind,
                     "gcv": [_f(v) for v in gcv],
                     "df": [_f(v) for v in dfs],
                     "i_min": i_min, "i_gcv": i_gcv},
            "delta_loo_r2": delta, "delta_loo_r2_se": se,
            "combined_loo_r2": comb_r2, "base_loo_r2": _f(base_r2),
            "captured_share": captured,
            "df_block": _f(df_z), "df_combined": df_comb,
            "sigma2_combined": sigma2_comb,
            "sigma2_base": _f(an0["sigma2_hat"]) if an0 is not None else None,
            "noise_guard": guard,
            "orthogonality_max": _f(orth_max),
            "hetero": bool(getattr(rec, "sample_weights", None) is not None),
            "notes": notes,
            "n_train": int(n_tr),
            "seconds": round(time.time() - t0, 2),
        }
        with self._lock:
            if (p.get("epoch") != self._epoch or self.stale
                    or self.result is not res
                    or self._residual_base_model is None):
                self._emit_done({"event": "residual_discarded",
                                 "reason": "fit changed"})
                return  # cleared/staled while rebuilding — discard
            self.view, self._yhat, self._xt = view, yhat, xt
            self.residual = block
        self._emit_done({"event": "residual_done",
                         "seconds": block["seconds"]})

    def _residual_sobol_readout(self, fitted, base_model,
                                names: List[str]) -> Optional[Dict[str, Any]]:
        """Total-order + first-order Sobol readout of the fitted residual ĝ
        and the combined f̂ = mean + ĝ (analysis §5c): Jansen/Saltelli
        pick-freeze on a scrambled-Sobol QMC design under the UNIFORM input
        measure on the model cube [0,1]^D — n(D+2) batched model evaluations
        via ``build_features``/``predict_batch`` only (backend-neutral).

        Model-based by necessity (total-order indices are not practically
        estimable from scattered data — conditioning on all-but-one variables
        is cursed in D and noise-biased), EXPLORATORY by label — never an
        admission criterion. BOTH normalizations are surfaced (within Var(ĝ)
        and against the model total Var(f̂)); which one leads is §6.8 expert
        material — the desk shows both and decides nothing. S_i(ĝ) > 0 is the
        per-variable orthogonality-defect diagnostic (in-sample projection vs
        the uniform measure) — expected, labeled, not a bug. The combined
        totals treat f̂ as one black box (no orthogonality assumption), so
        they are the honest headline; MC standard errors come from the same
        sample and are displayed."""
        from hifi_anova.analysis.qmc import sobol_cube_sample
        from hifi_anova.model.linear_residual import predict_residual_batch
        d = len(names)
        if d < 1:
            return None
        n = next(nv for dmax, nv in RES_SOBOL_N_BY_D if d <= dmax)
        pts = np.asarray(sobol_cube_sample(2 * d, n, seed=0), float)
        n = pts.shape[0]                 # sampler rounds up to a 2**m net
        A, B = pts[:, :d], pts[:, d:]

        def g_eval(x):
            return np.asarray(predict_residual_batch(fitted, x),
                              float).reshape(-1)

        def mean_eval(x):
            return np.asarray(base_model.mean_model.predict(
                base_model.build_phi1(x), base_model.build_phi2(x),
                base_model.build_phi3(x)), float).reshape(-1)

        gA, gB = g_eval(A), g_eval(B)
        fA, fB = mean_eval(A) + gA, mean_eval(B) + gB
        var_g = float(np.var(np.concatenate([gA, gB])))
        var_f = float(np.var(np.concatenate([fA, fB])))
        if not (var_g > 0 and var_f > 0):
            return None                  # ĝ ≈ 0 under the measure — nothing to read
        st_g: List = []
        st_g_se: List = []
        s1_g: List = []
        s1_g_se: List = []
        st_f: List = []
        st_f_se: List = []
        s1_f: List = []
        s1_f_se: List = []
        rt = float(np.sqrt(n))
        for i in range(d):
            if self._stop.is_set():
                return None
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            gi = g_eval(ABi)
            fi = mean_eval(ABi) + gi
            for ya, yb, yi, v, st, sts, s1, s1s in (
                    (gA, gB, gi, var_g, st_g, st_g_se, s1_g, s1_g_se),
                    (fA, fB, fi, var_f, st_f, st_f_se, s1_f, s1_f_se)):
                w = (ya - yi) ** 2                       # Jansen total-order
                st.append(_f(float(np.mean(w)) / (2.0 * v)))
                sts.append(_f(float(np.std(w, ddof=1)) / rt / (2.0 * v)))
                pr = yb * (yi - ya)                      # Saltelli-2010 first
                s1.append(_f(float(np.mean(pr)) / v))
                s1s.append(_f(float(np.std(pr, ddof=1)) / rt / v))
        return {
            "names": list(names), "n_qmc": int(n), "seed": 0,
            "measure": "uniform [0,1]^D (model space)",
            "estimator": "Jansen/Saltelli pick-freeze, scrambled-Sobol QMC",
            "var_g": _f(var_g), "var_f": _f(var_f),
            "g_share_of_f": _f(var_g / var_f),
            "st_g": st_g, "st_g_se": st_g_se,
            "s1_g": s1_g, "s1_g_se": s1_g_se,
            "st_f": st_f, "st_f_se": st_f_se,
            "s1_f": s1_f, "s1_f_se": s1_f_se,
            "label": ("model-based, uniform input measure, QMC ± MC error — "
                      "EXPLORATORY; not an admission criterion"),
        }

    def cmd_snapshot(self) -> Dict[str, Any]:
        return {"ok": True, **self.snapshot()}

    # ------------------------------------------------------------- worker
    def _fitting(self) -> bool:
        return self._active_job == "fit" or self._fit_queued

    def _mark_stale(self) -> None:
        self.stale = True
        self.gate = None  # a group-lasso selection is tied to one fitted design
        self.scan = None  # residual routing scores are tied to one fitted design too
        self.verify = None  # a Tier-III verification certifies ONE fitted model
        self.loo_test = None  # ΔLOO ranks are differences against ONE fitted model
        # the RESIDUAL bus block belongs to the fit it was computed on: detach
        # the attached residual too (model back to base; view is stale anyway)
        self._detach_residual()
        self._res_width_profile = None  # width profile is tied to one fit too
        self._emit({"event": "stale"})

    def _emit(self, ev: Dict[str, Any]) -> None:
        self.events.put(ev)

    def _emit_done(self, ev: Dict[str, Any]) -> None:
        """Queue a job-TERMINAL event (``*_done`` / ``cancelled`` / ``error``)
        for emission AFTER ``_job_loop``'s finally settles the job state.
        The ws server pushes a snapshot when it forwards one of these; a
        direct ``_emit`` raced the finally, so that snapshot could carry a
        stale busy flag (``residual_fitting`` etc.) with NO later snapshot —
        a remote waiter gating on the flag hung forever (Ses06 E2E repro).
        Worker-thread only; a cmd-path terminal event keeps plain _emit."""
        self._post_events.append(ev)

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._job_loop, daemon=True)
            self._worker.start()

    def _drain_jobs(self, kinds: set) -> set:
        """Remove queued jobs of the given kinds; return the kinds dropped."""
        kept, dropped = [], set()
        while True:
            try:
                item = self._jobs.get_nowait()
            except queue.Empty:
                break
            if item[0] in kinds:
                dropped.add(item[0])
            else:
                kept.append(item)
        for it in kept:
            self._jobs.put(it)
        return dropped

    def _cancel_warm(self) -> None:
        self._drain_jobs({"warm"})
        if self._active_job == "warm":
            self._stop.set()

    def _job_loop(self) -> None:
        while True:
            kind, payload = self._jobs.get()
            self._active_job = kind
            self._stop.clear()
            try:
                with self._abx():  # jobs run under the desk's array backend
                    if kind == "fit":
                        self._fit_worker()
                    elif kind == "verify":
                        self._verify_worker(payload)
                    elif kind == "lootest":
                        self._loo_test_worker(payload)
                    elif kind == "residual":
                        self._residual_worker(payload)
                    else:
                        self._warm_worker(payload)
            except Exception as exc:  # a job must never kill the worker
                self._emit_done({"event": "error",
                                 "message": f"{type(exc).__name__}: {exc}"})
                if kind == "fit":
                    with self._lock:
                        self.status = "ERROR"
                        self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._active_job = None
                self.warming = None
                if kind == "verify":
                    self._verify_pending = False
                elif kind == "lootest":
                    self._loo_test_pending = False
                elif kind == "residual":
                    self._residual_pending = False
                # terminal events flush ONLY now: a snapshot pulled on seeing
                # one reflects the settled job state (no stale busy flags)
                while self._post_events:
                    self._emit(self._post_events.pop(0))

    # -- shape keys: what a JAX recompile is keyed on (measured 2026-08-12:
    # (N, basis, hetero, uniform-vs-mixed path, total F, max K); mixed fits
    # with equal ΣF and max K reuse the compiled kernels regardless of WHICH
    # channel carries which K).
    def _shape_key(self, n: int, ks: List[int], basis: str, hetero: bool,
                   n_pairs: int = 0, k2: int = 0):
        try:
            from hifi_anova.core.features import basis_size  # flagged import
            F = int(sum(basis_size(int(k), True, basis) for k in ks))
        except Exception:
            F = int(sum(ks))
        # a pair fit adds order-2 columns (block size ~ basis_size(K2)² per pair) —
        # a genuinely new compile shape, so pairs/K2 join the key (uniform K forced)
        mixed = len(set(ks)) > 1 and not n_pairs
        return (int(n), str(basis), bool(hetero), mixed, F, int(max(ks)),
                int(n_pairs), int(k2)) + self._struct_key()

    def _struct_key(self):
        """The term-structure choices that change the design/compile shape and
        so must join the shape key (BR-06 order-selective membership; extended
        by the variance-side subset and per-pair orders as those land). Read
        from engine state — global to the fit, not per K-fader candidate."""
        return (tuple(sorted((nm, tuple(o)) for nm, o in self.term_orders.items())),
                tuple(sorted(self.var_muted)),  # BR-01 variance-side subset
                tuple(sorted((tuple(sorted(p)), k)  # BR-04 per-pair K2 overrides
                             for p, k in self.pair_k2_map.items())),
                tuple(sorted(tuple(sorted(p))      # K₂=0 pair mutes
                             for p in self.pair_muted)),
                (tuple(sorted(tuple(sorted(p)) for p in self.var_pairs)),  # BR-05
                 int(self.var_k2h)))

    def _fit_kwargs(self, ks: List[int], nms: List[str],
                    hetero: bool, progress=None,
                    pairs: Optional[List[int]] = None) -> Dict[str, Any]:
        """Build the ``hifi_anova`` kwargs. First-order by default; when ``pairs``
        (a list of active-position variable indices) is given, switch to a
        second-order fit over the clique among them (``mode='second'``, ``K2``,
        ``pair_selection`` = the active-variable list, ``lambda_order2``). The
        mixed per-variable-K path does NOT support pairs (backend
        ``NotImplementedError``), so a pair fit is always uniform K1=max — the
        caller surfaces that; here we simply omit ``basis_per_variable``."""
        kwargs: Dict[str, Any] = dict(
            K1=max(ks), K2=0, mode="first", variable_selection=None,
            verbose=False, seed=42, basis_name=self.basis,
            lambda_order1=float(self.lambda1),
            heteroscedastic=hetero,
            should_stop=self._stop.is_set,
            # numpy exact core: 'auto' resolves to numpy for every desk
            # config (they never touch the JAX-native residual/NN paths)
            backend=self.fit_backend,
        )
        if self.strategy != "auto":
            kwargs["strategy"] = self.strategy
        if progress is not None:
            kwargs["progress"] = progress
        if hetero:
            kwargs["lambda_h"] = float(self.lambda_h)
            # BR-01 variance-side mute — restrict the variance model to a subset
            vv = self._variance_variables_kwarg(nms)
            if vv is not None:
                kwargs["variance_variables"] = vv
        if pairs:
            kwargs["mode"] = "second"
            kwargs["lambda_order2"] = float(self.lambda2)
            k2map = self._pair_k2_mapping(nms)
            if k2map:
                # BR-04 per-pair K2: the mapping form PINS the exact pairs — no
                # pair_selection (mutually exclusive), no clique-induced extras.
                # Pair-excluded ([1]/[]) variables' pairs are already outside
                # the pin; pair-only ([2]) and noise-only ([]) memberships
                # compose here.
                kwargs["K2"] = k2map
                vo = {p: o for p, o
                      in self._variable_orders_kwarg(nms, hetero).items()
                      if o in ([2], [])}
                if vo:
                    kwargs["variable_orders"] = vo
            else:
                kwargs["K2"] = self._effective_k2(ks)
                kwargs["pair_selection"] = sorted(set(int(i) for i in pairs))
                # BR-06 order-selective membership — incl. noise-only ([])
                vo = self._variable_orders_kwarg(nms, hetero)
                if vo:
                    kwargs["variable_orders"] = vo
        elif len(set(ks)) > 1:
            # mixed per-variable K (first-order only — pairs force uniform K)
            kwargs["basis_per_variable"] = {
                j: {"basis": self.basis, "K": ks[j]} for j in range(len(nms))}
        if ("variable_orders" not in kwargs
                and "basis_per_variable" not in kwargs):
            # mean-EXCLUDED ([]) AND pair-only ([2]) memberships must reach
            # the backend even on a pair-less fit (the pair branches above
            # already carry them) — CONSTANT fits included since BR-12 (the
            # term-less channel stays an input for the COMPLEMENT; all-[] =
            # the intercept-only base). [2] on a pair-less fit = the
            # first-order block drops NOW (Ses07 mute semantics); [1] is
            # meaningless there (the fit is first-order anyway). The mixed
            # path can't (backend rejects term structure there) — membership
            # stays inert there, surfaced by the term summary.
            vo_none = {p: o for p, o
                       in self._variable_orders_kwarg(nms, hetero).items()
                       if o in ([2], [])}
            if vo_none:
                kwargs["variable_orders"] = vo_none
        # BR-05 second-order variance: a per-pair Sʰ term, hetero only,
        # endpoints variance-active. INDEPENDENT of the mean pair term — a
        # K₂=0-muted mean interaction may keep its noise interaction (the
        # backend validates variance pairs against variance variables only).
        if hetero and self.var_pairs and "basis_per_variable" not in kwargs:
            vsel = self._var_pairs_kwarg(nms)
            if vsel:
                kwargs["K2h"] = int(self.var_k2h)
                kwargs["var_pair_selection"] = vsel
        return kwargs

    def _pair_k2_mapping(self, nms: List[str]) -> Optional[Dict[tuple, int]]:
        """BR-04: ``{canonical(pos_i, pos_j): K2}`` over the ADMITTED pairs when
        any per-pair override is set — the K2-mapping form that pins the exact
        pairs. A pair with no override takes the global effective K2. Endpoints
        muted from the mean model, and pairs touching a marginal-only ([1])
        variable, are excluded (consistent with variable_orders). None when no
        override is set (keep the global scalar path). A K₂=0-MUTED admitted
        pair also forces the pinning form with itself omitted — the only way to
        exclude one pair of a clique (pair_selection would re-induce it)."""
        if not self.pair_k2_map and not self.pair_muted:
            return None
        pos = {nm: i for i, nm in enumerate(nms)}
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in nms]
        base = self._effective_k2(ks)
        # variables excluded from PAIRS: marginal-only ([1]) and noise-only ([])
        marg = {nm for nm in self.names
                if 2 not in self.term_orders.get(nm, [1, 2])}
        out: Dict[tuple, int] = {}
        has_live_pin = False
        for p in self.pairs:
            pr = tuple(sorted(p))
            if not set(pr) <= set(nms) or set(pr) & marg:
                continue
            if p in self.pair_muted:
                has_live_pin = True  # muted term: pin it OUT of the mapping
                continue
            i, j = pos[pr[0]], pos[pr[1]]
            key = (i, j) if i < j else (j, i)
            ov = self.pair_k2_map.get(self._pair_key(*pr))
            has_live_pin = has_live_pin or ov is not None
            out[key] = int(max(1, min(int(ov if ov else base), PAIR_K2_CAP)))
        # stay on the scalar/pair_selection path unless some override or mute
        # actually applies to a currently-admitted pair (a stale entry on a
        # removed pair must not silently switch the fit to the pinning form)
        return out if (out and has_live_pin) else None

    def _var_pairs_kwarg(self, nms: List[str]) -> Optional[List[tuple]]:
        """BR-05: canonical ``[(i, j), ...]`` active positions for the variance
        pairs that are ADMITTED (in self.pairs — a K₂=0-MUTED pair still
        qualifies: the mean and noise interaction terms are independent), both
        endpoints active and variance-active (not var_muted — the backend
        rejects a variance pair touching a variance-flat variable). None when
        nothing qualifies."""
        pos = {nm: i for i, nm in enumerate(nms)}
        admitted = {tuple(sorted(p)) for p in self.pairs}
        out = []
        for p in self.var_pairs:
            pr = tuple(sorted(p))
            if pr not in admitted:
                continue
            if not set(pr) <= set(nms) or (set(pr) & self.var_muted):
                continue
            i, j = pos[pr[0]], pos[pr[1]]
            out.append((i, j) if i < j else (j, i))
        return sorted(out) or None

    def _variance_variables_kwarg(self, nms: List[str]) -> Optional[List[int]]:
        """BR-01: sorted ACTIVE positions of the variables IN the variance model
        (active mean vars minus var_muted). None when the subset is the full
        active set (the default — no kwarg needed) or would be empty
        (defensive; the command already forbids emptying it)."""
        keep = [i for i, nm in enumerate(nms) if nm not in self.var_muted]
        if len(keep) == len(nms) or not keep:
            return None
        return sorted(keep)

    def _variable_orders_kwarg(self, nms: List[str],
                               hetero: bool) -> Dict[int, List[int]]:
        """BR-06: ``{active_pos: [orders]}`` for ACTIVE channels carrying a
        non-default order-selective membership. Empty for the default
        hierarchical model (every active channel at both orders).

        A pair-only ([2]) variable's first-order block is dropped
        IMMEDIATELY (Ses07 — the user-expected mute semantics; before, the
        entry was held back until an interaction touched the variable so it
        would not vanish from the model, a rationale BR-12 removed: a
        term-less channel honestly stays a COMPLEMENT input now). With no
        admitted interaction the variable carries NO mean term (surfaced as
        'pending'); once a pair touches it the pair term carries it
        (NON-HIERARCHICAL, disclosed).

        A mean-EXCLUDED ([]) variable is passed through on EVERY fit (BR-12):
        with HETERO its column feeds the variance model (noise-only); on a
        CONSTANT fit it is in neither model but stays an input the COMPLEMENT
        can capture (the backend accepts and discloses this since Ses06)."""
        pos = {nm: i for i, nm in enumerate(nms)}
        vo: Dict[int, List[int]] = {}
        for nm, orders in self.term_orders.items():
            if nm not in pos:
                continue
            o = sorted(orders)
            if o == [1, 2]:
                continue
            if o == []:
                vo[pos[nm]] = []
                continue
            vo[pos[nm]] = list(orders)
        return vo

    def _pair_touched_names(self, nms: List[str]) -> set:
        """Active variable names that have at least one ADMITTED interaction
        touching them (both endpoints active)."""
        out: set = set()
        active = set(nms)
        for p in self.pairs:
            pr = set(p)
            if pr <= active:
                out |= pr
        return out

    def _term_order_mode(self, nm: str) -> str:
        """BR-06 membership of a channel as a UI mode: 'both' (default),
        'pair' (order-2 only, non-hierarchical), 'marginal' (first-order,
        no interactions), or 'none' (mean-excluded — with HETERO the
        variance model keeps the column (noise-only); on a CONSTANT fit the
        channel is in neither model but stays a COMPLEMENT input, BR-12)."""
        o = sorted(self.term_orders.get(nm, [1, 2]))
        return {(1, 2): "both", (2,): "pair", (1,): "marginal",
                (): "none"}.get(tuple(o), "both")

    def _term_structure_view(self) -> Optional[Dict[str, Any]]:
        """BR-06 honesty summary for the view. For ACTIVE channels:
        ``nonhier`` = pair-only variables WITH an admitted interaction (the pair
        share absorbs any true marginal); ``pending`` = pair-only variables with
        NO interaction yet (Ses07: their first-order block is OUT of the fit
        NOW — the channel carries no mean term and stays a COMPLEMENT input
        until the user patches an interaction, e.g. via SCAN/ROUTE);
        ``marginal_only`` =
        first-order-only variables (interactions dropped); ``noise_only`` =
        mean-excluded variables tracked only by the variance model (HETERO);
        ``mean_excluded`` = mean-excluded variables on a CONSTANT fit — in
        NEITHER model, kept as COMPLEMENT inputs (BR-12; APPLIED, no longer
        inert); ``excluded_inert_mixed`` = 'none' channels a mixed
        per-channel-K fit cannot carry (the backend rejects term structure
        there — full membership applies, disclosed; Ses07: untouched
        pair-only channels join it — the mixed path cannot drop their
        first-order block either); ``intercept_only`` = True when EVERY
        active channel carries NO mean term (mean-excluded or untouched
        pair-only — the complement-only base, EXPLORATORY). None when the
        model is the default hierarchical one."""
        active = [nm for nm in self.names if nm not in self.muted]
        touched = self._pair_touched_names(active)
        # a mixed per-channel-K fit cannot carry variable_orders (mirrors
        # _fit_kwargs: pairs force uniform K, so mixed ⇔ unequal Ks, no pairs)
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in active]
        mixed = (len(set(ks)) > 1
                 and not self._active_pairs(active)[0])
        nonhier, marg, pending, nonly = [], [], [], []
        mexc, minert = [], []
        for nm in active:
            o = sorted(self.term_orders.get(nm, [1, 2]))
            if o == [2]:
                if mixed:
                    minert.append(nm)  # mixed-K can't carry [2] either
                elif nm in touched:
                    nonhier.append(nm)
                else:
                    pending.append(nm)
            elif o == [1] and self.pairs:
                marg.append(nm)
            elif o == []:
                if mixed:
                    minert.append(nm)
                elif self.hetero:
                    nonly.append(nm)
                else:
                    mexc.append(nm)
        if not (nonhier or marg or pending or nonly or mexc or minert):
            return None
        # channels with NO mean term: mean-excluded + untouched pair-only
        # (Ses07 — the first-order block drops immediately on uniform-K fits)
        n_off = len(nonly) + len(mexc) + len(pending)
        return {"nonhier": nonhier, "marginal_only": marg, "pending": pending,
                "noise_only": nonly, "mean_excluded": mexc,
                "excluded_inert_mixed": minert,
                "intercept_only": bool(active) and n_off == len(active)}

    def _active_pairs(self, nms: List[str]):
        """The admitted pairs whose BOTH endpoints are active, and the active-
        position variable indices they span. Returns ``(pair_name_tuples,
        endpoint_positions)`` — the endpoints are what the backend admits as a
        clique via ``pair_selection`` (so the fit may add induced pairs)."""
        pos = {nm: i for i, nm in enumerate(nms)}
        # a K₂=0-muted pair contributes no term and no endpoints — with every
        # pair muted the fit honestly degrades to first-order
        active = [tuple(sorted(p)) for p in self.pairs
                  if p <= set(nms) and p not in self.pair_muted]
        endpoints = sorted({pos[nm] for pr in active for nm in pr})
        return active, endpoints

    def _neighbor_jobs(self) -> List[Dict[str, Any]]:
        """Warm-fit payloads for every UNSEEN shape one K-fader step away.

        Shapes dedup by compile key, so from a uniform config the whole
        one-step neighborhood usually collapses to two jobs (all +1 moves
        share one shape, all -1 moves another). Last-touched channel first."""
        active_idx = [i for i, nm in enumerate(self.names) if nm not in self.muted]
        nms = [self.names[i] for i in active_idx]
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in nms]
        order = list(range(len(nms)))
        if self._last_touched in nms:
            j0 = nms.index(self._last_touched)
            order.remove(j0)
            order.insert(0, j0)
        out: List[Dict[str, Any]] = []
        planned: set = set()
        xs = self.X[:, active_idx]
        n = len(self.y)
        for j in order:
            for d in (1, -1):
                k2 = ks[j] + d
                if not (K_MIN <= k2 <= K_MAX):
                    continue
                ks2 = list(ks)
                ks2[j] = k2
                key = self._shape_key(n, ks2, self.basis, False)
                if key in self._seen_shapes or key in planned:
                    continue
                planned.add(key)
                out.append({"ks": ks2, "nms": nms, "xs": xs, "key": key,
                            "basis": self.basis, "epoch": self._epoch,
                            "label": "K=" + "·".join(map(str, ks2))})
                if len(out) >= WARM_MAX_JOBS:
                    return out
        return out

    def _warm_worker(self, p: Dict[str, Any]) -> None:
        """Pre-compile one neighboring shape. Never touches the view/result."""
        if (p["epoch"] != self._epoch or p["basis"] != self.basis
                or self.hetero or p["key"] in self._seen_shapes
                or self._stop.is_set()):
            return
        import hifi_anova as ha
        self.warming = p["label"]
        self._emit({"event": "warmup", "state": "start",
                    "message": f"warming shape {p['label']}"})
        try:
            kwargs = self._fit_kwargs(p["ks"], p["nms"], hetero=False)
            res = ha.hifi_anova(p["xs"], self.y, feature_names=p["nms"], **kwargs)
            try:
                # build-and-discard a view so the analytics kernels
                # (predict/curves/degrees) are warm too, not just the fit;
                # warn=False — a discarded warm view must not raise user warns
                active = [self.names.index(nm) for nm in p["nms"]]
                self._build_view(res, active, p["nms"], warn=False)
            except Exception:
                pass
            self._seen_shapes.add(p["key"])
            self._emit({"event": "warmup", "state": "done",
                        "message": f"shape {p['label']} warm"})
        except ha.HiFiCancelled:
            pass
        except Exception:
            pass  # warmup is best-effort by definition
        finally:
            self.warming = None

    def _fit_worker(self) -> None:
        import hifi_anova as ha
        with self._lock:
            self._fit_queued = False
            self.status = "FITTING"
        t0 = time.time()
        active = [i for i, nm in enumerate(self.names) if nm not in self.muted]
        nms = [self.names[i] for i in active]
        ks = [int(self.k.get(nm, DEFAULT_K)) for nm in nms]
        xs = self.X[:, active]
        # ROUTE (M3): admitted interactions among active channels. The backend
        # admits pairs as an active-variable clique, and mixed per-variable K is
        # unsupported with pairs — so a pair fit forces uniform K1=max. Surface
        # that coupling (DEC-045), never silently reshape.
        act_pairs, endpoints = self._active_pairs(nms)
        n_admitted = len(endpoints) * (len(endpoints) - 1) // 2 if endpoints else 0
        k2 = self._effective_k2(ks) if endpoints else 0
        self._pairs_forced_uniform_k = bool(endpoints) and len(set(ks)) > 1
        if self._pairs_forced_uniform_k:
            self._emit({"event": "warn",
                        "message": f"interactions force uniform K=max({max(ks)}) — "
                                   "mixed per-channel K is unsupported with pairs"})
        key = self._shape_key(len(self.y), ks, self.basis, self.hetero,
                              n_pairs=n_admitted, k2=k2)
        if key not in self._seen_shapes and self.fit_backend == "jax":
            # the COMPILING announcement is a JAX-backend fact (per-shape XLA
            # trace+compile); the numpy exact core has nothing to compile, so
            # announcing it there would be false latency honesty
            self._emit({"event": "compile",
                        "message": "new model shape — compiling (one-time)"})
        progress = lambda e: self._emit({"event": "progress",  # noqa: E731
                                         "stage": e.get("stage"),
                                         "fraction": _f(e.get("fraction")),
                                         "message": e.get("message")})
        kwargs = self._fit_kwargs(ks, nms, self.hetero, progress=progress,
                                  pairs=endpoints or None)
        try:
            try:
                res = ha.hifi_anova(xs, self.y, feature_names=nms, **kwargs)
            except ha.HiFiCancelled:
                raise
            except Exception:
                if "basis_per_variable" not in kwargs:
                    raise
                # mixed-basis path can reject some configs — fall back to uniform K
                kwargs.pop("basis_per_variable")
                self._emit({"event": "warn",
                            "message": "per-channel K rejected by backend — "
                                       "fitted with uniform K=max"})
                res = ha.hifi_anova(xs, self.y, feature_names=nms, **kwargs)
        except ha.HiFiCancelled:
            with self._lock:
                self.status = "FIT_READY" if self.result is not None else "READY"
            self._emit_done({"event": "cancelled"})
            return
        except Exception as exc:
            with self._lock:
                self.status = "ERROR"
                self.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            self._emit_done({"event": "error", "message": self.error})
            return

        try:
            view, yhat, xt = self._build_view(res, active, nms)
        except Exception as exc:
            with self._lock:
                self.status = "ERROR"
                self.error = f"view build failed: {type(exc).__name__}: {str(exc)[:300]}"
            self._emit_done({"event": "error", "message": self.error})
            return
        try:
            view["path"] = self._compute_path(res, nms)
        except Exception as exc:
            view["path"] = None
            self._emit({"event": "warn",
                        "message": f"λ path unavailable: {type(exc).__name__}: {exc}"})
        self._seen_shapes.add(key)
        with self._lock:
            # a fresh fit replaces the result: any residual belonged to the
            # OLD result (a stale-mark already detached it in the usual flow;
            # this covers a re-FIT with no settings change) — and so did the
            # width-profile cache (its epoch key does NOT change on a re-FIT)
            self.residual = None
            self._residual_base_model = None
            self._res_width_profile = None
            self.result = res
            self.view = view
            self._yhat = yhat
            self._xt = xt
            self._active_names = nms
            # record what ACTUALLY ran (kwargs may have lost basis_per_variable to
            # the mixed-K→uniform fallback above) for a faithful PRINT repro script
            self._last_fit_kwargs = {k: v for k, v in kwargs.items()
                                     if k not in ("should_stop", "progress")}
            self._last_active = list(active)
            self.stale = False
            self.error = None
            self.status = "FIT_READY"
            self.last_fit_seconds = time.time() - t0
            loo_r2 = view["master"].get("loo_r2")
            loo_nll = view["master"].get("loo_nll")
            if loo_r2 is not None or loo_nll is not None:
                self.loo_samples.append(
                    {"log_lambda1": _f(np.log10(self.lambda1)),
                     "loo_r2": loo_r2, "loo_nll": loo_nll,
                     "hetero": self.hetero})
                self.loo_samples = self.loo_samples[-200:]
        # default quick fit, 5..10 vars (auto_order2): this first-order fit just
        # produced the Sᶠ ranking — stage the budgeted TOP clique and tell the
        # client (its Tier-1 auto-refit loop picks it up, respecting the
        # AUTO-REFIT toggle; the fit is marked STALE either way, never silent)
        # INITIAL TRY, screened case: this first-order fit just produced the
        # Sᶠ ranking — admit the planned pair subset (pinned via the mapping
        # form) and tell the client (same order2_default flow as the d-rule)
        if self._auto_plan_pending:
            with self._lock:
                stage = (self.order2 == "off" and not self.pairs
                         and not self.stale and self.plan_speed != "off")
                self._auto_plan_pending = False
                plan = dict(self._plan) if self._plan else None
            if stage and plan and self._apply_plan_pairs(plan):
                self._mark_stale()
                self._emit({"event": "order2_default", "message":
                            (f"initial try: staged {plan['n_pairs']} of "
                             f"{plan['all_pairs']} pairs (K₂={plan['k2']}, "
                             "Sᶠ-heredity rank — heuristic first shot) — "
                             "refit to apply")})
        if self._auto_order2_pending:
            with self._lock:
                stage = (self.order2 == "off" and not self.pairs
                         and not self.stale)
                self._auto_order2_pending = False
            if stage and self._recompute_order2_top():
                self._mark_stale()
                t = self._order2_top
                self._emit({"event": "order2_default", "message":
                            ("default quick fit: staged "
                             + (f"TOP {t['m']} of {t['of']} channels' "
                                "2nd-order clique (Sᶠ rank, heuristic)"
                                if t else "ALL 2nd-order pairs")
                             + " — refit to apply")})
        self._emit_done({"event": "fit_done",
                         "seconds": round(self.last_fit_seconds, 2)})
        # warm-neighbor pre-compilation is a JAX-backend concept: the numpy
        # core has nothing to compile, so warming there is wasted work. Gated
        # HERE (not at server start) so a runtime BACKEND toggle to JAX gets
        # its warming back immediately.
        if (self.warm_neighbors and not self.hetero
                and self.fit_backend == "jax"):
            for job in self._neighbor_jobs():
                self._jobs.put(("warm", job))

    # ------------------------------------------------------------- λ path
    def _compute_path(self, res, nms: List[str]) -> Optional[Dict[str, Any]]:
        """Exact criterion path over λ₁ from the design the trainer solved.

        Re-solves the SAME penalized ridge (Phi, centered y, penalty shape
        reg_diag/λ₁ — the first-order penalty is linear in λ₁) on a log grid,
        via ``ridge_analytics``: exact plug-in LOO, σ̂²(λ) (noise-complexity
        curve), df, and per-channel core Sobol shares at every grid point.
        Homoscedastic fits only for now (a Stage-D fit is GLS-weighted and its
        unweighted path would be misleading guidance — deferred).

        Uses ``res._fitted_design`` (private, flagged for API promotion as the
        PathService enabler E1). Deviation from the plan's compute_reg_path:
        that helper assumes uniform per-variable blocks and reports GCV, while
        the desk needs mixed per-channel K and the exact-LOO convention that
        matches the master meter — ridge_analytics covers both.
        """
        rec = getattr(res, "_fitted_design", None)
        if rec is None or getattr(rec, "sample_weights", None) is not None:
            return None
        from hifi_anova.analysis.automl import ridge_analytics  # flagged import
        Phi = np.asarray(rec.Phi, float)
        yc = np.asarray(rec.y_centered, float)
        reg = np.asarray(rec.reg_diag, float)
        lam_fit = float(self.lambda1)
        if lam_fit <= 0 or not np.all(np.isfinite(reg)):
            return None
        # The λ₁ path varies ONLY the first-order penalty (which is linear in λ₁);
        # a second-order (pair) fit's pair block is penalized by λ₂ and must stay
        # FIXED as λ₁ sweeps — scaling the whole reg_diag by λ₁ would misrepresent
        # the pair penalty. Mask the first-order columns and scale just those.
        first_mask = np.zeros(len(reg), bool)
        # per-channel first-order column groups (uniform or mixed layout)
        groups: List[tuple] = []
        if getattr(rec, "sobol_groups", None) is not None:
            for order, gkey, cols, gram in rec.sobol_groups:
                if order == 1:
                    groups.append((nms[int(gkey)], cols, np.asarray(gram, float)))
                    first_mask[cols] = True
        else:
            b1 = rec.block(1)
            if b1 is None:
                return None
            first_mask[b1.columns] = True
            B = int(np.asarray(b1.gram).shape[0])
            start = b1.columns.start or 0
            for j, nm in enumerate(nms):
                groups.append((nm, slice(start + j * B, start + (j + 1) * B),
                               np.asarray(b1.gram, float)))

        grid = np.logspace(PATH_LOG10_MIN, PATH_LOG10_MAX, PATH_POINTS)
        if PATH_LOG10_MIN <= np.log10(lam_fit) <= PATH_LOG10_MAX:
            grid = np.unique(np.append(grid, lam_fit))
        var_y = float(np.var(self.y))
        n = len(yc)
        loo_mse = np.empty(len(grid))
        loo_se = np.empty(len(grid))
        sigma2 = np.empty(len(grid))
        dfs = np.empty(len(grid))
        aic = np.empty(len(grid))
        bic = np.empty(len(grid))
        gcv = np.empty(len(grid))
        shares = {nm: np.empty(len(grid)) for nm, _, _ in groups}
        for i, lam in enumerate(grid):
            pen = reg.copy()
            pen[first_mask] = reg[first_mask] * (lam / lam_fit)
            an = ridge_analytics(Phi, yc, pen)
            loo_mse[i] = an["loo_cv"]
            sq = np.asarray(an["loo_residuals"]) ** 2
            loo_se[i] = float(np.std(sq)) / np.sqrt(n)
            sigma2[i] = an["sigma2_hat"]
            dfs[i] = an["df"]
            aic[i] = an["aic"]
            bic[i] = an["bic"]
            gcv[i] = an["gcv"]
            w = np.asarray(an["w"])
            var_i = {nm: max(float(w[cols] @ G @ w[cols]), 0.0)
                     for nm, cols, G in groups}
            tot = sum(var_i.values())
            for nm, _, _ in groups:
                shares[nm][i] = var_i[nm] / tot if tot > 0 else 0.0
        i_min = int(np.argmin(loo_mse))
        # 1-SE pick: largest λ whose LOO is within one SE of the minimum
        within = np.nonzero(loo_mse <= loo_mse[i_min] + loo_se[i_min])[0]
        i_1se = int(within[-1]) if len(within) else i_min
        i_s2 = int(np.argmin(sigma2))
        loo_r2 = (1.0 - loo_mse / var_y) if var_y > 0 else np.zeros_like(loo_mse)
        return {
            "log10_lambda": [_f(v) for v in np.log10(grid)],
            "loo_r2": [_f(v) for v in loo_r2],
            "sigma2": [_f(v) for v in sigma2],
            "df": [_f(v) for v in dfs],
            "shares": {nm: [_f(v) for v in arr] for nm, arr in shares.items()},
            "i_min": i_min, "i_1se": i_1se,
            "sigma2_min": _f(sigma2[i_s2]),
            "log10_lambda_sigma2_min": _f(np.log10(grid[i_s2])),
            # Ses07: information-criterion curves along the SAME λ₁ grid (the
            # λ-bank criteria, not GATE's group-lasso BIC path) + their minima,
            # so the desk can show where the criteria disagree with exact LOO.
            "aic": [_f(v) for v in aic],
            "bic": [_f(v) for v in bic],
            "gcv": [_f(v) for v in gcv],
            "i_aic": int(np.argmin(aic)),
            "i_bic": int(np.argmin(bic)),
            "i_gcv": int(np.argmin(gcv)),
        }

    # ------------------------------------------------------------- payload
    def _build_view(self, res, active, nms, warn: bool = True):
        from hifi_anova.analysis.component_eval import frequency_decomposition
        ms = res.sobol["mean_sobol"]["first_order"]
        # 𝓕·core normalization (share of the FITTED variance incl. the residual
        # model) — the SPECTRUM normalization toggle reads it; core stays the
        # headline everywhere else
        ms_tot = (res.sobol.get("mean_sobol_total", {}) or {}).get(
            "first_order", {}) or {}
        lv_all = res.sobol.get("log_variance_sobol") or {}
        lv = lv_all.get("first_order", {}) if isinstance(lv_all, dict) else {}
        var_y = float(np.var(self.y))
        r2c = _f(res.r_squared_classical) or _f(res.r_squared) or 0.0
        fid = res.fidelity if isinstance(res.fidelity, dict) else {}
        fval = _f(fid.get("value", 1.0)) or 1.0
        calib = bool(res.noise_scale_is_calibration)
        # loo_cv is on the whitened scale after Stage D — LOO-R² only for homo
        loo_r2 = None
        if res.loo_cv is not None and not calib:
            loo_r2 = _f(1.0 - float(res.loo_cv) / var_y)
        xs = self.X[:, active]
        yhat = np.asarray(res.predict(xs), float).reshape(-1)
        try:
            xt = (np.asarray(res.transformer.transform(xs), float)
                  if res.transformer is not None else xs)
        except Exception:
            xt = xs

        channels = []
        for j, i in enumerate(active):
            nm = nms[j]
            ci = None
            if isinstance(res.sobol_ci, dict) and nm in res.sobol_ci:
                s, lo, hi = res.sobol_ci[nm]
                ci = [_f(lo), _f(hi)]
            try:
                cx, cy = res.component_curve(nm, n_points=41)
                curve = [[_f(v) for v in np.asarray(cx, float)],
                         [_f(v) for v in np.asarray(cy, float)]]
            except Exception:
                curve = None
            try:
                fd = frequency_decomposition(res.model, j)
                deg = [_f(v) for _, v in
                       sorted(fd.items(), key=lambda kv: int("".join(
                           c for c in kv[0] if c.isdigit()) or 0))]
            except Exception:
                deg = None
            channels.append({
                "name": nm, "muted": False,
                "sf": _f(ms.get(j, 0.0)),
                "sf_total": _f(ms_tot.get(j)) if ms_tot else None,
                "sh": _f(lv.get(j)) if lv else None,
                "ci": ci, "k": int(self.k.get(nm, DEFAULT_K)),
                "curve": curve, "degrees": deg,
                "order": self._term_order_mode(nm),  # BR-06
                "var_muted": nm in self.var_muted,   # BR-01
            })
        for nm in self.names:
            if nm in self.muted:
                channels.append({"name": nm, "muted": True, "sf": None,
                                 "sf_total": None,
                                 "sh": None, "ci": None,
                                 "k": int(self.k.get(nm, DEFAULT_K)),
                                 "curve": None, "degrees": None,
                                 "order": self._term_order_mode(nm),
                                 "var_muted": nm in self.var_muted})

        # pair strips (C15): second-order Sobol per ADMITTED interaction, read from
        # the fit's ground-truth pair_indices (positions in the active order). We
        # mark which pairs the desk requested vs which the clique closure induced,
        # so a disjoint-pair patch is honest about the extra terms it pulled in.
        pairs_view = self._pair_strips(res, nms)

        segs = [{"label": c["name"],
                 "share": _f(max(c["sf"], 0.0) * fval * r2c)}
                for c in channels if not c["muted"] and c["sf"]]
        # Ses08 (watch-list #1): admitted interactions belong in the MAIN ledger
        # too — previously their variance appeared only in the by-degree bar, so
        # the two bars disagreed on a pair fit. Same accounting as the channels
        # (share of model variance × fidelity × R²).
        segs += [{"label": f"{p['a']}×{p['b']}",
                  "share": _f(max(p["sf"], 0.0) * fval * r2c)}
                 for p in pairs_view if p.get("sf")]
        resid_share = _f(max(ms.get("residual", 0.0)
                             if isinstance(ms, dict) else 0.0, 0.0))
        ledger = {
            "vars": segs,
            "structured": _f(max(0.0, (1.0 - fval) * r2c)),
            "noise": _f(max(0.0, 1.0 - r2c)),
        }
        # parity payload: observed vs predicted, colored by predicted noise
        rng = np.random.default_rng(1)
        nidx = rng.permutation(len(self.y))[: min(600, len(self.y))]
        sig = None
        if self.hetero:
            try:
                s2 = np.asarray(res.sigma_x2(self.X[:, active][nidx]), float)
                if np.ptp(s2) > 1e-12:  # constant => guard skipped Stage D
                    sig = [_f(v) for v in np.sqrt(np.maximum(s2, 0.0))]
            except Exception:
                sig = None
        # calibration QQ statistics for the SAME parity subsample: standardized
        # residuals r/σ̂ (pointwise σ̂(x) when the fit has one, else the global
        # σ̂) plus the N(0,1) order quantiles. Computed HERE so the client only
        # sorts/draws — the inverse-normal CDF no longer lives in JS.
        qq = None
        try:
            resid = self.y[nidx] - yhat[nidx]
            if sig is not None:
                sd = np.array([v if (v is not None and v > 1e-12) else np.nan
                               for v in sig], float)
            else:
                s0 = _f(res.sigma_hat)
                sd = np.full(len(nidx), s0 if (s0 and s0 > 1e-12) else np.nan)
            sd = np.where(np.isfinite(sd), sd, 1.0)
            z = resid / sd
            from scipy.stats import norm
            q = norm.ppf((np.arange(len(z)) + 0.5) / len(z))
            qq = {"z": [_f(v) for v in z], "q": [_f(v) for v in q]}
        except Exception as exc:
            qq = None
            if warn:
                self._emit({"event": "warn", "message":
                            "calibration QQ unavailable: "
                            f"{type(exc).__name__}: {exc}"})
        # layered parity (Ses05/Ses06): ŷ per model layer so the desk can look
        # at the parity cloud one effect at a time. The mean model is additive —
        # f0 + φ₁·w₁ [+ φ₂·w₂] (+ complement ĝ) — so every layer is an exact
        # slice, computed straight off ``res.model`` (the COMBINED model when a
        # complement is attached; its ``mean_model`` is still the structured
        # mean, so the structured slices exclude ĝ and ``active`` == res.predict
        # folds it in).
        #
        # A FIXED, always-emitted button set (Ses06, user): the desk sees a
        # STABLE row — 1st / 2nd / compl / 1+2 / active — that GREYS OUT when a
        # layer is not available for this fit, rather than appearing/disappearing
        # context-sensitively. Two kinds:
        #   ISOLATED (kind "iso"): each order ALONE on the y-scale = f0 + that
        #     one component (1st / 2nd / compl). Separate, not added.
        #   COMBINED (kind "cum"): 1+2 (structured mean) and ``active`` = the
        #     full active model (== res.predict, the rightmost/default).
        # ``active`` keeps its usual master.r2 / +bus badge; the other layers are
        # peels with their own IN-SAMPLE R² — a descriptive cloud readout (can be
        # NEGATIVE for an isolated slice — honest), NOT a selection criterion or
        # a new default.
        layers = None
        # f0 = the intercept-only (no-fit) baseline; surfaced so the client can
        # anchor the parity decomposition-ladder overlay's horizontal baseline
        # (Ses08). Defaults None so it survives the layered-peel except path.
        f0_out = None
        try:
            model = res.model
            mm = getattr(model, "mean_model", None)
            if mm is not None:
                # build_phi* operate in the model's TRANSFORMED space, the same
                # space res.predict feeds — which CLIPS to [0,1] (api.py
                # _transform_new_inputs: quantile-boundary saturation). Feeding
                # the unclipped xt would desync the layers from the plotted ŷ on
                # any out-of-range row; match res.predict's clip exactly.
                xtc = np.clip(np.asarray(xt, float), 0.0, 1.0)
                p1 = model.build_phi1(xtc)
                p2 = model.build_phi2(xtc)   # None when no admitted pairs
                p3 = model.build_phi3(xtc)   # None when no triples
                has2 = (p2 is not None and int(getattr(mm, "K2", 0)) > 0
                        and len(mm.w2) > 0)
                has_c = getattr(model, "residual_net", None) is not None
                # BR-12: an intercept-only mean (fo_included=()) has NO
                # first-order block — grey the 1st layer instead of showing a
                # flat f0 line that pretends to be a marginal fit
                fo_fit = getattr(model, "fo_included", None)
                has1 = not (fo_fit is not None and len(fo_fit) == 0)
                f0 = float(np.asarray(mm.f0, float).reshape(-1)[0])
                f0_out = _f(f0)
                c1 = np.asarray(p1, float) @ np.asarray(mm.w1, float)  # φ₁·w₁
                c2 = (np.asarray(p2, float) @ np.asarray(mm.w2, float)
                      if has2 else None)                               # φ₂·w₂
                gap = (yhat - np.asarray(mm.predict(p1, p2, p3), float
                                         ).reshape(-1)) if has_c else None  # ĝ
                sst = var_y * len(self.y)

                def _entry(key, label, kind, avail, yl):
                    # a FIXED-position layer — always emitted so the desk sees a
                    # STABLE button set (Ses06: greyed when not available for
                    # this fit, never appearing/disappearing). ŷ/R² only when
                    # available; the R² is in-sample over all rows (descriptive,
                    # can be negative for an isolated slice — honest).
                    ysub = r2 = None
                    if avail and yl is not None:
                        yl = np.asarray(yl, float).reshape(-1)
                        sse = float(np.sum((self.y - yl) ** 2))
                        r2 = _f(1.0 - sse / sst) if sst > 1e-300 else None
                        ysub = [_f(v) for v in yl[nidx]]
                    return {"key": key, "label": label, "kind": kind,
                            "available": bool(avail), "yhat": ysub, "r2": r2}

                # the canonical LAYER set (gui3 EFFECTS exposes 1st/2nd/compl —
                # no 3rd-order mean; a 3rd term, if ever present, rides inside
                # ``active`` and is excluded from ``1+2`` by construction).
                layers = [
                    _entry("i1", "1st", "iso", has1,
                           (f0 + c1) if has1 else None),
                    _entry("i2", "2nd", "iso", has2,
                           (f0 + c2) if has2 else None),
                    _entry("ic", "compl", "iso", has_c,
                           (f0 + gap) if has_c else None),
                    _entry("o12", "1+2", "cum", has1 and has2,
                           (f0 + c1 + c2) if (has1 and has2) else None),
                    # the full active model — reuses res.predict (yhat) bit-exact
                    _entry("active", "active", "cum", True, yhat),
                ]
        except Exception as exc:
            layers = None
            if warn:
                self._emit({"event": "warn", "message":
                            "layered parity unavailable: "
                            f"{type(exc).__name__}: {exc}"})
        parity = {"y": [_f(v) for v in self.y[nidx]],
                  "yhat": [_f(v) for v in yhat[nidx]],
                  "sigma": sig, "qq": qq, "layers": layers, "f0": f0_out}

        # empirical coverage of the central noise intervals — the backend
        # calibration_report convention: NOISE scale only (no epistemic term),
        # so a high-df fit overstates coverage (tooltip says so). Pass/warn
        # verdicts use ±2 binomial SE, computed HERE not in JS. On a whitened
        # Stage-D fit with no pointwise σ̂ the scale would be wrong — refuse
        # (None) instead of standardizing by the wrong σ̂.
        #
        # BR-07 (DEC-055): coverage is measured on the HELD-OUT rows
        # (val ∪ test from ``res.split_indices``) — rows the fit never saw — so
        # the LEDs report out-of-sample calibration, not the in-sample
        # optimism that a high-df model manufactures on its own train split.
        # Only when the split leaves nothing held out (rare) do we fall back to
        # all rows, and the payload says which via ``rows``.
        calibration = None
        try:
            sd_all = None
            scale = None
            if self.hetero:
                try:
                    s2a = np.asarray(res.sigma_x2(xs), float)
                    if np.ptp(s2a) > 1e-12:
                        sd_all = np.sqrt(np.maximum(s2a, 1e-300))
                        scale = "pointwise"
                except Exception:
                    sd_all = None
            if sd_all is None and not calib:
                s0 = _f(res.sigma_hat)
                if s0 and s0 > 1e-12:
                    sd_all = np.full(len(self.y), float(s0))
                    scale = "global"
            if sd_all is not None:
                # held-out row ids (authoritative, BR-07); empty → in-sample
                rows = "held-out"
                try:
                    si = res.split_indices
                    held = np.concatenate([np.asarray(si["val"], int),
                                           np.asarray(si["test"], int)])
                except Exception:
                    held = np.arange(len(self.y))
                if held.size == 0:
                    held = np.arange(len(self.y))
                    rows = "in-sample"
                z_all = ((self.y - yhat) / sd_all)[held]
                from scipy.stats import norm
                n_all = len(z_all)
                lv_rows = []
                for a in (0.5, 0.9, 0.95, 0.99):
                    zc = float(norm.ppf((1 + a) / 2))
                    covg = float(np.mean(np.abs(z_all) <= zc))
                    tol = 2.0 * float(np.sqrt(a * (1 - a) / n_all))
                    lv_rows.append({"level": a, "coverage": _f(covg),
                                    "ok": bool(abs(covg - a) <= tol),
                                    "tol": _f(tol)})
                calibration = {"n": int(n_all), "scale": scale,
                               "rows": rows, "levels": lv_rows}
            else:
                # no usable noise scale — surface WHY instead of a silent blank
                # (Ses08 watch-list: whitened-fit calibration was a silent None)
                calibration = {"refused": True, "levels": [], "reason": (
                    "whitened Stage-D fit exposes no pointwise σ̂(x) — residuals "
                    "cannot be standardized to check interval coverage"
                    if calib else "no usable noise scale (σ̂) on this fit")}
        except Exception as exc:
            calibration = {"refused": True, "levels": [],
                           "reason": f"unavailable ({type(exc).__name__})"}
            if warn:
                self._emit({"event": "warn", "message":
                            "calibration coverage unavailable: "
                            f"{type(exc).__name__}: {exc}"})

        # Criterion / per-point LOO analytics of the FITTED design (Ses07 LOO
        # surfacing). One extra ridge solve with the same instrument api.py's
        # ``result.loo()`` uses — profiled intercept and GLS weights respected —
        # so AIC/BIC/GCV/ess and the per-point leverage (hat diagonal H_ii) /
        # deleted-residual arrays share the reported LOO convention exactly.
        #
        # NB the trainer fits on a TRAIN SPLIT (``preprocess_data`` shuffles and
        # holds out val/test rows), so the per-point arrays live on the train
        # rows — they go into a separate ``diag`` payload, NOT ``parity`` (which
        # plots predictions over ALL rows). The mapping back to dataset rows is
        # reconstructed from the fit seed and VERIFIED against y before use;
        # when it cannot be verified, row ids / σ̂ are omitted, never guessed.
        crit = {"aic": None, "bic": None, "gcv": None, "ess_per_param": None}
        diag = None
        rec = getattr(res, "_fitted_design", None)
        if rec is not None:
            try:
                from hifi_anova.analysis.automl import ridge_analytics  # flagged
                from hifi_anova.training.fitted_design import (
                    MEAN_INTERCEPT_PROFILED_JOINT_GLS)
                profiled = (getattr(rec, "mean_intercept_mode", None)
                            == MEAN_INTERCEPT_PROFILED_JOINT_GLS)
                y_ana = (rec.y_centered + rec.f0) if profiled else rec.y_centered
                an = ridge_analytics(rec.Phi, y_ana, rec.reg_diag,
                                     weights=rec.sample_weights,
                                     profile_intercept=profiled)
                crit = {"aic": _f(an["aic"]), "bic": _f(an["bic"]),
                        "gcv": _f(an["gcv"]),
                        "ess_per_param": _f(an["ess_per_param"])}
                lev = np.asarray(an["leverages"], float)
                lr = np.asarray(an["loo_residuals"], float)
                resid = np.asarray(an["residuals"], float)
                y_tr = np.asarray(rec.y_centered, float) + float(rec.f0)
                yhat_tr = y_tr - resid  # both intercept modes: resid is in y units
                n_tr = len(y_tr)
                # authoritative train-row mapping (BR-07, DEC-055):
                # ``res.split_indices['train']`` are the original dataset rows
                # the fit trained on, in Phi order — no seed reconstruction, no
                # y-verify guessing. Kept behind a try so an older result (no
                # split_indices) degrades to "no row ids" instead of raising.
                train_idx = None
                try:
                    ti = np.asarray(res.split_indices["train"], int)
                    if len(ti) == n_tr:
                        train_idx = ti
                except Exception:
                    train_idx = None
                if train_idx is None and warn:
                    self._emit({"event": "warn", "message":
                                "train-row mapping unavailable — per-point "
                                "views show no dataset row ids"})
                sub = np.random.default_rng(2).permutation(n_tr)[: min(600, n_tr)]
                sig_tr = None
                if self.hetero and train_idx is not None:
                    try:
                        s2t = np.asarray(
                            res.sigma_x2(self.X[:, active][train_idx[sub]]), float)
                        if np.ptp(s2t) > 1e-12:  # constant => guard skipped Stage D
                            sig_tr = [_f(v) for v in np.sqrt(np.maximum(s2t, 0.0))]
                    except Exception:
                        sig_tr = None
                diag = {
                    "y": [_f(v) for v in y_tr[sub]],
                    "yhat": [_f(v) for v in yhat_tr[sub]],
                    "leverage": [_f(v) for v in lev[sub]],
                    "loo_resid": [_f(v) for v in lr[sub]],
                    "sigma": sig_tr,
                    "row": ([int(train_idx[i]) for i in sub]
                            if train_idx is not None else None),
                    "n_train": int(n_tr),
                    # 2·df/N high-leverage rule of thumb — the SAME df as the
                    # leverage array (Σ hat diag = df), computed here so the
                    # client draws the line instead of re-deriving it
                    "leverage_threshold": (_f(2.0 * float(an["df"]) / n_tr)
                                           if n_tr else None),
                }
            except Exception as exc:
                diag = None
                if warn:
                    # a systematic analytics failure must not hide behind "—"
                    # (Ses07 watch-list #7)
                    self._emit({"event": "warn", "message":
                                "criteria/per-point diagnostics unavailable: "
                                f"{type(exc).__name__}: {exc}"})

        # R14 detail (Ses07): the Tier-II regularity flags BEHIND the single
        # tier2_ok boolean, so a red guarantee lamp can say why. Only exists when
        # the fit actually reported Tier II (a Stage-D fit with a variance design).
        tier2_detail = None
        if res.loo_tier2_guarantee_holds is not None:
            try:
                j = res.loo(tier=2)  # O(N·F_h²), cheap — public API
                if j.get("loo_tier") == 2:
                    tier2_detail = {
                        "h_rcond": _f(j.get("h_rcond")),
                        "n_correction_clipped": int(j.get("n_correction_clipped") or 0),
                        "variance_floor_active": bool(j.get("loo_variance_floor_active")),
                        "hessian_ill_conditioned": bool(
                            j.get("variance_hessian_ill_conditioned")),
                        "correction_clipped": bool(j.get("loo_nll_correction_clipped")),
                    }
            except Exception as exc:
                tier2_detail = None
                if warn:
                    self._emit({"event": "warn", "message":
                                "Tier-II flag detail unavailable: "
                                f"{type(exc).__name__}: {exc}"})

        # BR-12: the FITTED mean is intercept-only (fo_included=(), no solved
        # pair block) — the structured decomposition is empty and any
        # explanatory power lives in the COMPLEMENT (EXPLORATORY). The client
        # renders a fixed banner off this flag; not stylable away by mode.
        _fo_fit = getattr(res.model, "fo_included", None)
        intercept_only = bool(
            _fo_fit is not None and len(_fo_fit) == 0
            and not (res.model.pair_indices is not None
                     and int(getattr(res.model.mean_model, "K2", 0)) > 0
                     and len(res.model.mean_model.w2) > 0))
        master = {
            "r2": _f(res.r_squared), "r2_classical": r2c, "loo_r2": loo_r2,
            "loo_nll": _f(res.loo_nll),
            "press": None if (res.loo_cv is None or calib)
                     else _f(float(res.loo_cv) * len(self.y)),
            "df": _f(res.df), "df_residual": _f(res.df_residual),
            "sigma_hat": _f(res.sigma_hat),
            "sigma_is_calibration": bool(res.noise_scale_is_calibration),
            "fidelity": fval, "hetero": self.hetero,
            "loo_tier": int(res.loo_tier or 1),
            "tier2_ok": None if res.loo_tier2_guarantee_holds is None
                        else bool(res.loo_tier2_guarantee_holds),
            "var_y": _f(var_y),
            "noise_floor_share": _f(max(0.0, 1.0 - r2c)),
            # A1 (R17): max |corr| in the MODEL's transformed space over the
            # ACTIVE channels — the space where the Sobol independence
            # assumption actually applies (raw-scale corr differs under the
            # per-column quantile map). Pre-fit `dataset.max_corr` is the raw
            # fallback the client labels differently.
            "max_corr": _max_abs_corr(np.asarray(xt, float), list(nms)),
            # R18: efficient−interpretable Sobol gap (hetero two-fit convention);
            # None on a homoscedastic fit (no gap surface exists).
            "sobol_gap_max": _sobol_gap_max(res),
            # R16 leg from the result itself: the backend flags a fit whose
            # structure it selected on the same data (e.g. auto interaction
            # discovery). NB a desk-supplied explicit pair_selection is NOT flagged
            # here (api.py) — the engine's session-selection tracking covers that.
            "structure_selected_on_same_data": bool(
                (getattr(res, "inference_metadata", None) or {}).get(
                    "structure_selected_on_same_data", False)),
            # Ses07 LOO surfacing: information criteria of THIS fitted design
            # (NB distinct from GATE's group-lasso BIC path — these ride the λ
            # bank, not the variable-selection path) + Tier-II flag detail.
            "aic": crit["aic"], "bic": crit["bic"], "gcv": crit["gcv"],
            "ess_per_param": crit["ess_per_param"],
            "tier2_detail": tier2_detail,
            "intercept_only": intercept_only,
        }
        return ({"channels": channels, "ledger": ledger, "master": master,
                 "parity": parity, "diag": diag, "pairs": pairs_view,
                 "calibration": calibration,
                 "pairs_mixed_k": bool(self._pairs_forced_uniform_k),
                 "term_structure": self._term_structure_view(),  # BR-06
                 "residual_first_order_share": resid_share}, yhat, xt)

    def _pair_strips(self, res, nms: List[str]):
        """Second-order Sobol per admitted interaction, from the fit's real
        ``pair_indices``. Core share = ``mean_sobol['second_order'][(i,j)]``;
        total share = ``mean_sobol_total['second_order']``. Each strip is flagged
        ``requested`` (the desk clicked it) or induced by the clique closure.
        K₂=0-MUTED pairs are absent from the fitted model but keep a strip
        (``muted`` row, fader at 0) so the mute stays visible and reversible."""
        try:
            lv_muted = (res.sobol.get("log_variance_sobol", {}) or {}).get(
                "second_order", {}) or {}
        except Exception:
            lv_muted = {}
        try:
            pidx = getattr(res.model, "pair_indices", None)
            if pidx is None:
                return self._muted_pair_rows(nms, set(), lv_muted)
            pidx = np.asarray(pidx)
            if pidx.size == 0:
                return self._muted_pair_rows(nms, set(), lv_muted)
        except Exception:
            return self._muted_pair_rows(nms, set(), lv_muted)
        so = res.sobol.get("mean_sobol", {}).get("second_order", {}) or {}
        so_tot = (res.sobol.get("mean_sobol_total", {}) or {}).get("second_order", {}) or {}
        # BR-05: per-pair log-variance Sobol (second order) — populated only on a
        # hetero fit with variance pairs that survived the Stage-D guard
        lv_so = (res.sobol.get("log_variance_sobol", {}) or {}).get(
            "second_order", {}) or {}
        requested = {tuple(sorted(p)) for p in self.pairs}
        # BR-04: the global effective K2 each pair falls back to without an override
        base_k2 = self._effective_k2(
            [int(self.k.get(nm, DEFAULT_K)) for nm in nms] or [DEFAULT_K])
        out = []
        for pk, row in enumerate(pidx):
            i, j = int(row[0]), int(row[1])
            if i >= len(nms) or j >= len(nms):
                continue
            a, b = nms[i], nms[j]
            key = (i, j) if (i, j) in so else (j, i)
            name_key = tuple(sorted((a, b)))
            # coarse top-view thumbnail of the ORTHOGONAL component f̂ₐᵦ — lets a
            # pair strip preview the pure interaction term the way a first-order
            # scribble previews f̂ᵢ (zero-centered color scale, like solo_pair).
            # Capped to the first few pairs so the fit isn't slowed down.
            thumb = None
            if pk < PAIR_THUMB_MAX:
                try:
                    _g, z = self._pair_surface(res, i, j, G=PAIR_THUMB_G)
                    zl = self._sym_zlim(z)
                    thumb = {"z": [[_f(v) for v in r] for r in z],
                             "zmin": _f(-zl), "zmax": _f(zl)}
                except Exception:
                    thumb = None
            ov = self.pair_k2_map.get(self._pair_key(a, b))  # BR-04 override
            sh_key = key if key in lv_so else (j, i) if (j, i) in lv_so else key
            out.append({
                "a": a, "b": b,
                "sf": _f(max(float(so.get(key, 0.0)), 0.0)),
                "sf_total": _f(so_tot.get(key)) if so_tot else None,
                "requested": name_key in requested,
                "thumb": thumb,
                # per-pair harmonic order actually used, and whether it's a
                # user override vs the global default (BR-04)
                "k2": int(ov) if ov else int(base_k2),
                "k2_override": ov is not None,
                # BR-05: per-pair variance term — requested (var_on) and the
                # measured log-variance Sobol Sʰ (None until it survives a fit)
                "var_on": self._pair_key(a, b) in self.var_pairs,
                "sh": _f(max(float(lv_so.get(sh_key, 0.0)), 0.0)) if lv_so else None,
                # K₂=0 mute — True only on a stale view (a muted pair leaves
                # the fitted model on refit and moves to _muted_pair_rows)
                "muted": self._pair_key(a, b) in self.pair_muted,
            })
        out += self._muted_pair_rows(
            nms, {tuple(sorted((q["a"], q["b"]))) for q in out}, lv_so)
        # requested pairs first, then induced; strongest within each group
        out.sort(key=lambda p: (not p["requested"], -(p["sf"] or 0.0)))
        return out

    def _muted_pair_rows(self, nms: List[str], seen: set,
                         lv_so=None) -> List[Dict[str, Any]]:
        """Strip rows for K₂=0-muted requested pairs that are NOT in the fitted
        MEAN model (``seen`` = name-pairs already emitted): no mean term, no
        Sᶠ₂, no thumbnail — the visible, reversible fader-at-zero state. The
        VARIANCE term is independent of the mean mute, so a muted pair whose
        noise interaction survived the fit still shows its measured Sʰ₂
        (``lv_so`` = the fit's second-order log-variance Sobol by position)."""
        base_k2 = self._effective_k2(
            [int(self.k.get(nm, DEFAULT_K)) for nm in nms] or [DEFAULT_K])
        pos = {nm: i for i, nm in enumerate(nms)}
        rows = []
        for p in sorted(self.pair_muted, key=lambda q: tuple(sorted(q))):
            pr = tuple(sorted(p))
            if pr in seen or not set(pr) <= set(nms) or p not in self.pairs:
                continue
            ov = self.pair_k2_map.get(p)
            sh = None
            if lv_so:
                i, j = pos[pr[0]], pos[pr[1]]
                skey = ((i, j) if (i, j) in lv_so
                        else (j, i) if (j, i) in lv_so else None)
                if skey is not None:
                    sh = _f(max(float(lv_so.get(skey, 0.0)), 0.0))
            rows.append({"a": pr[0], "b": pr[1], "sf": None, "sf_total": None,
                         "requested": True, "thumb": None,
                         "k2": int(ov) if ov else int(base_k2),
                         "k2_override": ov is not None,
                         "var_on": p in self.var_pairs, "sh": sh,
                         "muted": True})
        return rows

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "catalog": REGISTRY,
                # control limits — the client builds its fader/slider ranges from
                # these, so UI ranges cannot drift from what the engine accepts
                "limits": {
                    "k_min": K_MIN, "k_max": K_MAX,
                    "lambda_log10_min": PATH_LOG10_MIN,
                    "lambda_log10_max": PATH_LOG10_MAX,
                    "lambda_hard_min": LAMBDA_MIN, "lambda_hard_max": LAMBDA_MAX,
                    "pair_k2_cap": PAIR_K2_CAP,
                    "order2_all_max_cols": ORDER2_ALL_MAX_COLS,
                    "csv_max_n": CSV_MAX_N,
                    # C16 RESIDUAL bus fader/path ranges
                    "residual_families": list(RESIDUAL_FAMILIES),
                    "residual_m_max": RESIDUAL_M_MAX,
                    "residual_lambda_log10_min": RES_PATH_LOG10_MIN,
                    "residual_lambda_log10_max": RES_PATH_LOG10_MAX,
                    "residual_width_log10_min": RES_WIDTH_LOG10_MIN,
                    "residual_width_log10_max": RES_WIDTH_LOG10_MAX,
                    "residual_kernels": list(RES_KERNELS),
                },
                # interpretive thresholds — the client styles from these; the
                # numbers themselves are engine (tested) claims, not JS literals
                "thresholds": {
                    "corr_warn": CORR_WARN, "gap_warn": GAP_WARN,
                    "role_sf": ROLE_SF_MIN, "role_sh": ROLE_SH_MIN,
                    "ktrim_tail": KTRIM_TAIL,
                    "pred_sh_weight": PRED_SH_WEIGHT,
                    "noise_agree_spread": NOISE_AGREE_SPREAD,
                },
                "noise_triangulation": self._noise_triangulation(),
                "dataset": {"label": self.dataset_label,
                            "id": self.dataset_id,
                            "names": list(self.names),
                            "n": 0 if self.y is None else int(len(self.y)),
                            "max_corr": self.max_corr,
                            "noise_model_free": self.noise_model_free,
                            "quality": list(self.data_quality)},
                "settings": {
                    "k": dict(self.k), "muted": sorted(self.muted),
                    "lambda1": _f(self.lambda1), "lambda_h": _f(self.lambda_h),
                    "lambda2": _f(self.lambda2), "pairs": self._pairs_as_lists(),
                    # EFFECTS order-2 mode, live projected column count, TOP
                    # preselection detail, and the moving quick-fit budget —
                    # all engine claims; the client only renders them
                    **self._order2_snapshot_fields(),
                    "pair_k2": self.pair_k2, "pair_k2_cap": PAIR_K2_CAP,
                    # BR-04 per-pair K2 overrides: [[a, b, K2], ...]
                    "pair_k2_map": sorted(
                        [sorted(p) + [k] for p, k in self.pair_k2_map.items()]),
                    # K₂-fader-at-zero pair mutes: [[a, b], ...]
                    "pair_muted": sorted(sorted(p) for p in self.pair_muted),
                    # resolved pair fidelity (auto = min(max active K, cap)) —
                    # computed HERE so the client renders it instead of
                    # re-deriving _effective_k2 in JS
                    "pair_k2_effective": self._effective_k2(
                        [int(self.k.get(nm, DEFAULT_K)) for nm in self.names
                         if nm not in self.muted] or [DEFAULT_K]),
                    "basis": self.basis, "strategy": self.strategy,
                    "hetero": self.hetero, "auto_reset": self.auto_reset,
                    "auto_order2": self.auto_order2,
                    # INITIAL TRY speed preset (off = planner disabled)
                    "plan_speed": self.plan_speed,
                    # array backend fits run on ('auto' → numpy exact core)
                    "fit_backend": self.fit_backend,
                    # BR-06 order-selective membership: name -> UI mode string
                    "term_orders": {nm: self._term_order_mode(nm)
                                    for nm in self.term_orders},
                    # BR-13: every active channel mean-excluded — the STAGED
                    # intercept-only base (the 1st-INDIVIDUAL rocker's state;
                    # the fitted-model flag is view.master.intercept_only)
                    "intercept_only_staged": self._intercept_only_staged(),
                    # BR-01 variance-side mute: names asserted variance-flat
                    "var_muted": sorted(self.var_muted),
                    # BR-05 second-order variance: pairs carrying a noise term + order
                    "var_pairs": sorted(sorted(p) for p in self.var_pairs),
                    "var_k2h": int(self.var_k2h),
                    # C16 RESIDUAL bus control state (family + fader values;
                    # None = library default / auto-λ) — TAKES/PROFILE carry it
                    "residual": dict(self.residual_cfg),
                },
                "status": self.status, "stale": self.stale,
                "error": self.error,
                "last_fit_seconds": self.last_fit_seconds,
                "warming": self.warming,
                "view": self.view, "scan": self.scan, "gate": self.gate,
                "selection": self._selection_state(),  # R16 CONFIG-CONDITIONAL
                "verify": self.verify,                 # Tier-III oracle result
                "verifying": (self._active_job == "verify"
                              or self._verify_pending),
                "loo_test": self.loo_test,             # R25 rank-only ΔLOO
                "loo_testing": (self._active_job == "lootest"
                                or self._loo_test_pending),
                # C16 RESIDUAL bus: the fitted block (None = nothing attached)
                # + job state; the CONTROL state rides in settings.residual
                "residual": self.residual,
                "residual_fitting": (self._active_job == "residual"
                                     or self._residual_pending),
                "loo_samples": list(self.loo_samples),
            }
