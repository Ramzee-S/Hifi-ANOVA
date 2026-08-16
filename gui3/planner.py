"""INITIAL TRY — first-shot structure planner (budget-ladder heuristic).

Plans the structure of a QUICK FIRST fit from (N rows, D variables, Z max
columns) alone — O(1), no data touched. A parameter budget

    B = min(Z, N // gamma)          (gamma = min rows per column, ridge-safe)

is spent on model upgrades in a fixed priority ladder, cheap-and-valuable
before expensive-and-marginal:

    base     intercept + K=1 mains        (mandatory; ridge keeps it solvable
                                           even when B can't afford it)
    k1:2     quadratic mains
    pairs    bilinear pair terms (K2=1)   greedy count, "all" or "screened"
    k1:3     cubic mains
    k1:4     quartic mains
    k2:2     upgrade chosen pairs to K2=2
    k1:5     quintic mains

Degradation is monotone: a big budget buys the whole ladder (bounded by the
per-D caps below), a tiny one degrades to a plain linear model. The "screened"
pair mode only says *how many* pairs fit the budget — WHICH pairs is the
caller's job (the console ranks by the fitted first-order Sf, heredity-style,
and stages that after the first fit; this module never sees data).

This is a pragmatic compute-time heuristic for the very first shot, NOT a
reviewed statistical default-selection rule (A.6 brief material — same class
as the TOP budget and the d-rule). The console flags every screened choice in
the R16 selection state. Costs are stated in design columns via a caller-
supplied basis-size function, so Fourier/Haar count honestly (Haar grows as
2^K − 1); the module itself imports nothing from the backend.
"""
from __future__ import annotations

from typing import Callable, Dict, List


def default_k1_cap(d: int) -> int:
    """First-order fidelity cap for a FIRST fit — even with budget to burn,
    don't open wilder than this (the faders remain for deliberate moves)."""
    if d <= 4:
        return 5
    if d <= 7:
        return 4
    if d <= 12:
        return 3
    return 2


def default_k2_cap(d: int) -> int:
    """Pair-order cap: tensor-product quadratic pair blocks are affordable at
    small D only; beyond that a first shot stays bilinear (K2=1)."""
    return 2 if d <= 5 else 1


# ladder rungs after the mandatory base (intercept + K=1 mains)
LADDER = ("k1:2", "pairs", "k1:3", "k1:4", "k2:2", "k1:5")


def plan_first_try(n: int, d: int, z: int,
                   basis_cols: Callable[[int], int],
                   *,
                   gamma: int = 3,
                   min_pairs: int = 3,
                   k1_max: int = 5,
                   k2_max: int = 2,
                   k1_cap: Callable[[int], int] = default_k1_cap,
                   k2_cap: Callable[[int], int] = default_k2_cap,
                   ladder=LADDER) -> Dict[str, object]:
    """Plan the first-shot structure. ``basis_cols(k)`` = columns of one
    variable's first-order block at order k (a pair block costs
    ``basis_cols(k2)**2``). Returns a JSON-safe dict:

    ``{k1, k2, n_pairs, all_pairs, pair_mode: none|all|screened,
       cols, budget, binding, notes: [str, ...]}``

    ``cols`` counts intercept + mains + pair blocks at the planned orders;
    ``binding`` names what limited the model (``rows``/``speed``/``caps``/
    ``underdetermined``).
    """
    n, d, z = int(n), int(d), int(z)
    if d < 1 or n < 1 or z < 1:
        raise ValueError("plan_first_try needs n, d, z >= 1")
    b1 = {k: int(basis_cols(k)) for k in range(1, max(2, k1_max) + 1)}
    budget = min(z, n // max(1, gamma))
    k1_top = min(k1_cap(d), k1_max)
    k2_top = min(k2_cap(d), k2_max)
    all_pairs = d * (d - 1) // 2
    notes: List[str] = []

    plan = {"k1": 1, "k2": 0, "n_pairs": 0, "all_pairs": all_pairs,
            "pair_mode": "none", "budget": budget}
    spent = 1 + d * b1[1]  # mandatory base: intercept + K=1 mains
    if budget < spent:
        # even the linear model is over budget — fit it anyway (ridge keeps it
        # solvable), but say so instead of silently screening variables away
        plan.update(cols=spent, binding="underdetermined")
        notes.append(
            f"budget {budget} is below even the linear model "
            f"({spent} columns) — fitted with ridge regularization; "
            "more rows (or a slower preset) would be needed for a "
            "determined first shot")
        plan["notes"] = notes
        return plan

    exhausted = False
    for rung in ladder:
        left = budget - spent
        if left <= 0:
            exhausted = True
            break
        if rung.startswith("k1:"):
            target = min(int(rung.split(":")[1]), k1_top)
            # buy one order at a time so a partial raise stays affordable
            while plan["k1"] < target:
                step = d * (b1[plan["k1"] + 1] - b1[plan["k1"]])
                if step > left:
                    exhausted = True
                    break
                plan["k1"] += 1
                spent += step
                left -= step
        elif rung == "pairs":
            if d >= 2 and k2_top >= 1 and plan["n_pairs"] == 0:
                per_pair = b1[1] ** 2  # K2=1 (bilinear) pair block
                affordable = left // per_pair
                take = min(all_pairs, int(affordable))
                if take >= min(min_pairs, all_pairs):
                    plan["k2"] = 1
                    plan["n_pairs"] = take
                    spent += take * per_pair
                    if take < all_pairs:
                        plan["pair_mode"] = "screened"
                        notes.append(
                            f"budget admits {take} of {all_pairs} pairs — "
                            "rank by the fitted first-order Sf (heredity); "
                            "a pair can matter with weak mains, SCAN covers "
                            "the rest")
                    else:
                        plan["pair_mode"] = "all"
                else:
                    exhausted = True
        elif rung.startswith("k2:"):
            target = int(rung.split(":")[1])
            if plan["n_pairs"] and target <= k2_top and target > plan["k2"]:
                extra = (b1[target] ** 2 - b1[plan["k2"]] ** 2) * plan["n_pairs"]
                if extra <= left:
                    plan["k2"] = target
                    spent += extra
                else:  # no partial/mixed upgrade — keep the pair blocks regular
                    exhausted = True

    plan["cols"] = spent
    if not exhausted and plan["k1"] >= k1_top and (
            plan["n_pairs"] in (0, all_pairs)) and (
            plan["k2"] in (0, k2_top)):
        plan["binding"] = "caps"
    else:
        plan["binding"] = "speed" if z <= n // max(1, gamma) else "rows"
    plan["notes"] = notes
    return plan
