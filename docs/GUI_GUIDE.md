# HiFi Console (gui3) — User Guide

*A local browser desk for interactive HiFi-ANOVA fitting and diagnostics.*

> **Status: experimental / pre-alpha.** The console is under active development
> and ships **as source** in the alpha (not in the installed wheel). It is a
> front-end over the public `hifi_anova` library — every number it shows is
> computed by the same library documented in [`USER_GUIDE.md`](USER_GUIDE.md).
> **Nothing the console displays is a blessed model selection**: the parity
> ladder is descriptive, the COMPLEMENT bus is exploratory, and the selection
> aids (AUTO, LOO-test, GATE) *propose* — they never auto-decide. See
> [Honesty & scope](#honesty--scope).

---

## Table of contents

1. [What it is](#1-what-it-is)
2. [Install & run](#2-install--run)
3. [The desk at a glance](#3-the-desk-at-a-glance)
4. [A first session](#4-a-first-session)
5. [Loading data](#5-loading-data)
6. [Fitting: faders, basis, and the backend](#6-fitting-faders-basis-and-the-backend)
7. [Muting effects (mean / variance)](#7-muting-effects-mean--variance)
8. [Interactions: SCAN → ROUTE](#8-interactions-scan--route)
9. [Reading the fit: the monitor scopes](#9-reading-the-fit-the-monitor-scopes)
10. [The parity ladder (LAYER peel)](#10-the-parity-ladder-layer-peel)
11. [The COMPLEMENT bus](#11-the-complement-bus)
12. [Diagnostics & honesty lamps](#12-diagnostics--honesty-lamps)
13. [Selection aids (propose-only)](#13-selection-aids-propose-only)
14. [TAKES & PRINT](#14-takes--print)
15. [Honesty & scope](#honesty--scope)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What it is

The HiFi Console presents a HiFi-ANOVA model as a **mixing desk**: each input
variable is a channel with faders (basis order `K`, penalties), mute buttons,
and solo; the master section carries the global penalties and the
homoscedastic/heteroscedastic switch; and a monitor deck shows the Sobol
spectrum, parity, the regularization path, and a variance ledger. You tune the
model and it re-fits interactively, streaming progress over a WebSocket.

It is a **thin front-end**: the console never re-implements the statistics. Fits,
Sobol indices, confidence intervals, leave-one-out, and the residual/complement
projection all run through the public library API, so what you see on the desk
matches what you would get from a script.

## 2. Install & run

The console needs the `gui3` extra (FastAPI + uvicorn), installed from a source
checkout:

```bash
pip install -e ".[gui3]"          # fastapi + uvicorn[standard]
python -m gui3.server             # default http://127.0.0.1:8630
```

Then open **http://127.0.0.1:8630** in a browser. Use **one browser tab per
server** (a single fit worker backs each server process).

Server options:

```bash
python -m gui3.server \
  --port 8630 \                   # listen port
  --n 4000 \                      # demo sample size
  --no-warmup \                   # skip the startup background fit
  --fit-backend auto              # auto|numpy|jax  (auto => NumPy exact core)
```

**Startup.** Unless `--no-warmup` is given, a heteroscedastic *fatigue* demo is
loaded and a background fit absorbs the one-time JAX compile (~20 s). After that,
warm homoscedastic refits are sub-second; a heteroscedastic (Stage-D) fit takes
a few seconds and streams progress.

> The console is a local developer tool. It binds to `127.0.0.1` and has no
> authentication — do not expose the port to a network.

## 3. The desk at a glance

The window is organized into four regions:

- **Transport bar (top).** FIT / cancel, AUTO-REFIT, the first-shot planner
  (`1st TRY`), AUTO tuning proposals, TAKES, PRINT, RESET ALL, the mode strip
  (MIX / ROUTE / DEEP / TAKES), and the **BACKEND CORE\|JAX** chip.
- **Channel rack (left).** One strip per variable: the `K` fader, basis
  selector, `M` (mean-effect mute), variance mutes, SOLO, and the per-channel
  Sobol readout. A header strip carries whole-variable in/out, sort order, and a
  NAME / #ID label toggle.
- **Master rail (right).** Global penalties (`λ₁`, `λ_h`), the **HOMO / HETERO**
  switch (real Stage-D), the honesty lamps, and the LOO verification tools.
- **Monitor deck (center/right).** The scope selector — SPECTRUM, SLICE,
  λ-PATH, LEDGER — plus PARITY (with its five views and the LAYER peel), the
  ROUTING matrix, and the docked **COMPLEMENT bus**.

## 4. A first session

1. **Start the server** and open the page. The fatigue demo is already loaded and
   fitted (after the warmup).
2. **Watch the SPECTRUM scope** — the Sobol indices for the current fit, one bar
   per variable (mean effect, and in a heteroscedastic fit the log-variance
   index too).
3. **Move a `K` fader** on a channel and let it re-fit — the spectrum and PARITY
   update. On the NumPy core a refit is ~0.1 s.
4. **Toggle HETERO** to fit the variance model (Stage D); the calibration and
   LOO-tier lamps light up.
5. **Run SCAN** (in ROUTE mode) to score candidate second-order interactions,
   then click a cell to patch a pair in.
6. **Check the honesty lamps** on the master rail before reading anything as a
   conclusion (see [§12](#12-diagnostics--honesty-lamps)).

## 5. Loading data

- **Demo datasets.** The server loads a registry dataset on startup (`fatigue` by
  default). Re-loading uses the same interactivity sample cap.
- **CSV import.** Upload a numeric CSV (one header row, numeric cells). The
  importer infers the target column (`y` / `target` / the last column unless you
  name one), drops rows with missing selected values (reported), and
  row-subsamples large files to the interactivity cap. Inputs are installed at
  raw scale — the fit's quantile transformer maps them into the model's `[0,1]`
  space, so SCAN, SOLO, and σ̂ readouts stay correct.

Loading a genuinely new dataset resets per-channel `K`, mutes, admitted pairs,
and returns the basis to the Legendre default (basis is never carried across
datasets).

## 6. Fitting: faders, basis, and the backend

- **`K` faders** set the per-variable basis order. Bases can be **mixed per
  variable**; if the backend rejects a particular mixed configuration the console
  falls back to the uniform max-`K` path and says so.
- **Basis selector** — **LEGENDRE / FOURIER / HAAR** per channel (Gram matrices
  from near-diagonal to identity; see the library guide's
  [Bases & effect signatures](USER_GUIDE.md#7-bases--effect-signatures)).
- **Penalties** — `λ₁` (mean ridge) and, in a heteroscedastic fit, `λ_h` (the
  variance penalty) on the master rail.
- **AUTO-REFIT** re-fits on every change; turn it off to stage several changes
  and fit once.
- **The backend chip — BACKEND CORE\|JAX.** `CORE` is the **NumPy exact core**
  (float64), the default under `--fit-backend auto`; it removes the per-shape JAX
  recompile so a fader move re-fits in ~0.1 s. `JAX` uses the JAX path (a
  COMPILING banner appears only there). The two agree to machine precision on
  homoscedastic fits (see [USER_GUIDE §2](USER_GUIDE.md#2-installation--quickstart)
  for the library-level `backend=` parameter).

## 7. Muting effects (mean / variance)

Each channel's **`M`** button (teal) mutes **one** effect at a time:

- **On** → the first-order mean effect is in the fit.
- **1°-muted** → the first-order mean block is dropped from the fit *immediately*
  (its Sobol mean index goes to zero), while the variable stays available for
  interactions. This is the same as pulling the `K` fader to zero.
- **NOISE-ONLY** (heteroscedastic fits) → the mean is fully excluded but the
  variable is kept in the variance model.

Muting **every** channel's first-order effect with no interactions admitted
honestly lands on an **intercept-only base** (`f₀` only, R² ≈ 0) — the desk says
so plainly rather than posing a flat line as a fit. With a pair admitted, the
all-muted state is a **pure second-order fit**. Variance-side mutes work
analogously for the log-variance model.

Colored = the effect is in the fit; grey = muted.

## 8. Interactions: SCAN → ROUTE

Switch the mode strip to **ROUTE** to work with second-order interactions:

- **SCAN** scores the missing pairs (`scan_missing_pairs`) and flags the top
  candidates. It is tied to the current fit — it clears when the fit goes stale
  and re-scans after a fit while you are in ROUTE.
- The **ROUTING matrix** shows every variable pair. **Click any cell** to patch
  that pair in (a second-order refit) or remove it. Patched cells are marked,
  induced cells (clique closure) are distinguished, and candidate cells glow.
- Muted variables remain visible as greyed dead rows ("unmute to patch"); a
  patched pair touching a muted endpoint shows as dormant.

## 9. Reading the fit: the monitor scopes

The scope selector switches the main monitor:

- **SPECTRUM** — the Sobol indices of the current fit (the headline
  interpretable attribution; on a heteroscedastic fit, the mean *and*
  log-variance spectrum).
- **SLICE** — a component curve for the soloed channel.
- **λ-PATH** — the regularization path: Sobol (or fit quality) across the `λ₁`
  grid, with the exact-LOO minimum and 1-SE marks, and optional **CRIT**
  overlays (AIC / BIC / GCV) so you can see where the criteria's minima disagree
  with exact LOO.
- **LEDGER** — a Var(y) accounting: how much variance each effect explains,
  split by interaction order, with an optional per-basis-degree bar. An
  "unsplit" bucket appears when a channel lacks a degree split (never guessed).

**SOLO** on a channel overlays its component curve and partial residuals (and
σ̂(xᵢ) on a heteroscedastic fit).

## 10. The parity ladder (LAYER peel)

The **PARITY** scope plots predicted vs. observed, with an R² / LOO-R² readout,
and its toggle cycles five views:

- **PARITY** (pred vs. observed), **RESID**, **LOO-RESID** (out-of-sample
  residual `r/(1−Hᵢᵢ)`), **CAL-QQ** (calibration), and **LEVERAGE** (hat diagonal
  vs. predicted, with the `2·df/N` rule-of-thumb line; high-leverage rows
  highlighted).

In parity mode the **LAYER** rocker peels the fit by contribution — a fixed
five-layer set: **1st** (first-order only), **2nd** (second-order only),
**compl** (the complement, if fitted), **1+2** (cumulative), and **active** (the
full shipped prediction). Each isolated layer shows its own in-sample R² when
available.

> The ladder is **descriptive** — a way to *see* how much each order contributes.
> An isolated-layer R² can be negative (honest for a single slice) and is **never
> a selection number**.

## 11. The COMPLEMENT bus

The **COMPLEMENT bus** (docked under PARITY) fits a post-hoc, orthogonal residual
model on top of the **current** structured fit — a flexible smooth
(NYSTRÖM / RBF / RFF) that captures structure the admitted terms did not.

- Because the complement is projected **orthogonally** to the fitted design, it
  **cannot change the structured fit's attribution** (this is tested). The
  master meters keep the structured fit's Sobol numbers; only the PARITY label
  updates to the *combined* model (marked `+bus`).
- Controls: the family, an M-features ladder and WIDTH mini-faders, and a `λ_res`
  fader with the engine's criterion curve (proposed exact-LOO minimum, GCV
  overlay, applied playhead). `FIT COMPLEMENT ▶` runs it; `✕` detaches.
- The LEDGER's structured segment then splits captured-by-complement vs.
  still-unexplained, and (on homoscedastic fits) reports a paired ΔLOO-R².

> The complement is **exploratory**. It is a way to ask "how much structure is
> left?" — not an admitted part of the model, and its `λ_res` / width are not
> blessed defaults. A complement job always ends in exactly one outcome:
> attached, discarded (with a reason), or error.

## 12. Diagnostics & honesty lamps

The master rail carries lamps that tell you when a reading needs caution:

- **CONFIG-CONDITIONAL** — lights when structure was *chosen on this data* (a
  mute, a patched pair) rather than fixed up front; the intervals are then
  conditional on that in-session selection.
- **Two-fit gap** — surfaces `sobol_gap` (the efficient/GLS minus the
  interpretable/unit-weight Sobol) on a heteroscedastic fit; a homoscedastic fit
  carries no gap (never a fabricated number).
- **LOO tier** — states which leave-one-out tier is in force and, in red, *why*
  a tier-2 guarantee is at risk (variance floor binds, ill-conditioning,
  corrections clipped), pointing to **VERIFY LOO** as the authority.
- **Calibration** — the whitened-σ̂ calibration lamp on a Stage-D fit.

Two on-demand LOO tools:

- **VERIFY LOO** runs the exact nested-refit oracle (`exact_loo_nll`) through the
  job queue and compares it to the reported closed-form tier; it refuses
  honestly on homoscedastic fits (where the tiers already coincide).
- **Master criteria readouts** — AIC / BIC / GCV / N-per-df from the real fitted
  design.

## 13. Selection aids (propose-only)

These help you explore configurations but **never decide** for you:

- **`1st TRY`** — a first-shot planner (OFF / FAST / BAL / THOR) that budgets a
  sensible initial design (mains → screened pairs → higher orders). Disclosed as
  exploratory.
- **AUTO** — reads the current fit's diagnostics and *proposes* concrete changes
  (a `λ₁` toward the min-LOO / 1-SE mark, per-channel `K` trims where the top
  degree carries < 3 % energy), each with a one-line rationale and APPLY /
  APPLY ALL. It never auto-applies.
- **LOO TEST** — ranks candidate variable drops / pair adds by paired ΔLOO-R²
  (± paired SE) through real refits. It is **rank-only**: there is no threshold,
  keep/drop verdict, or auto-apply — the stopping rule is deliberately withheld
  pending review.

## 14. TAKES & PRINT

- **TAKES** capture the current *fit* (distinct from a PROFILE, which is just
  settings): save, A/B compare, and a LOO-R²-vs-df leaderboard. Takes persist
  per browser (localStorage).
- **PRINT** exports a report: a Markdown summary, a **runnable repro script**
  rebuilt from the actual fit kwargs (registry datasets get their loader), and
  the library's `res.summary()` captured verbatim — copy or download.

## Honesty & scope

The console is deliberately built so that exploration never masquerades as a
blessed result:

- **No blessed selections.** The parity ladder is descriptive; the COMPLEMENT bus
  is exploratory; AUTO, `1st TRY`, LOO-TEST and GATE *propose* and never
  auto-decide; `λ_res` / width and the LOO-test stopping rule are gated pending
  expert sign-off.
- **Attribution is protected.** The complement is orthogonal by construction and
  cannot move the structured fit's Sobol numbers.
- **Honest limits are shown, not hidden.** Intercept-only bases, at-risk LOO
  tiers, config-conditional intervals, and mixed-K refusals are surfaced with
  lamps and reasons rather than silently worked around.

For the underlying statistics — what a Sobol index, a two-fit gap, a LOO tier,
or a residual model *means* — see the library [`USER_GUIDE.md`](USER_GUIDE.md).

## 16. Troubleshooting

- **The page is blank / won't connect.** Confirm the server is running and you
  opened the printed `http://127.0.0.1:<port>` URL; use one tab per server.
- **First fit is slow.** The one-time JAX compile runs at startup (~20 s). Switch
  the **BACKEND** chip to `CORE` (the default) to avoid per-shape recompiles.
- **"fit running — cancel first".** A load or a new fit was requested mid-fit —
  cancel the running fit (the ■ button) first.
- **A mixed-basis configuration was rejected.** The console falls back to the
  uniform max-`K` path and discloses it; simplify the per-variable mix if you
  need the exact configuration.
- **A COMPLEMENT didn't attach.** It reports a reason (fit changed, cancelled, no
  train-row mapping, instrument mismatch, view-rebuild failure); re-fit and try
  again on a settled fit.
