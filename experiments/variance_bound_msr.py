"""Empirical variance of the SHAPIQ MSR estimator under leverage sampling.

Validates Theorem `var_msr`: for i.i.d. leverage sampling
P(T) = 1 / ((n+1) * C(n, |T|)), the SII MSR estimator obeys
    Var <= O(||v||_inf^2 * log n / |T|)         if |S| = 1
    Var <= O(||v||_inf^2 * n^(|S|-1) / |T|)     if |S| >= 2

Uses SHAPIQLeverageTruncated to extract estimates at all budgets from a single
sample batch per seed (cumsum trick). One i.i.d. with-replacement sample stream
covers every budget cutoff we need.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _msr_leverage_truncated import SHAPIQLeverageTruncated
from shapiq_games.benchmark.local_xai.benchmark_tabular import Independent


def make_estimator(n: int, max_order: int, seed: int) -> SHAPIQLeverageTruncated:
    sampling_weights = np.ones(n + 1, dtype=float)
    return SHAPIQLeverageTruncated(
        n=n,
        max_order=max_order,
        index="SII",
        sampling_weights=sampling_weights,
        pairing_trick=False,
        random_state=seed,
    )


def estimate_v_inf(game, n: int, n_samples: int, rng: np.random.Generator) -> float:
    coals = rng.integers(0, 2, size=(n_samples, n)).astype(bool)
    extras = np.zeros((2, n), dtype=bool)
    extras[1] = True
    coals = np.vstack([extras, coals])
    vals = np.asarray(game(coals))
    return float(np.max(np.abs(vals - vals[0])))


def run(out_path: Path, n_seeds: int, max_order: int, x_index: int) -> None:
    print(f"[setup] loading IndependentLinear60 game (x={x_index})")
    game = Independent(
        x=x_index,
        model_name="decision_tree",
        imputer="marginal",
        normalize=True,
        random_state=42,
    )
    n = int(game.n_players)
    print(f"[setup] n_players = {n}")

    rng = np.random.default_rng(0)
    v_inf = estimate_v_inf(game, n, n_samples=5000, rng=rng)
    print(f"[setup] estimated ||v||_inf = {v_inf:.4f}")

    budgets = np.unique(np.geomspace(200, 20_000, 12).astype(int))
    print(f"[setup] budgets = {budgets.tolist()}")

    probe = make_estimator(n=n, max_order=max_order, seed=0)
    interactions = [S for S in probe.interaction_lookup if 1 <= len(S) <= max_order]
    interactions.sort(key=lambda s: (len(s), s))
    n_int = len(interactions)
    sizes = np.array([len(s) for s in interactions], dtype=np.int32)
    print(f"[setup] tracking {n_int} interactions of order 1..{max_order}")

    phi = np.zeros((n_seeds, len(budgets), n_int), dtype=np.float64)
    pos_for_S = np.array(
        [probe.interaction_lookup[S] for S in interactions], dtype=np.int64
    )

    t0 = time.perf_counter()
    for s_idx in range(n_seeds):
        seed_t0 = time.perf_counter()
        est = make_estimator(n=n, max_order=max_order, seed=s_idx)
        per_budget = est.approximate_at_budgets(budgets=list(budgets), game=game)
        for b_idx, budget in enumerate(budgets):
            phi[s_idx, b_idx, :] = per_budget[int(budget)][pos_for_S]
        if (s_idx + 1) % max(1, n_seeds // 20) == 0:
            elapsed = time.perf_counter() - t0
            per_seed = elapsed / (s_idx + 1)
            eta = per_seed * (n_seeds - s_idx - 1)
            print(
                f"[run] seed {s_idx + 1}/{n_seeds}  last={time.perf_counter() - seed_t0:.1f}s  "
                f"avg={per_seed:.1f}s  ETA={eta / 60:.1f} min"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        phi=phi,
        sizes=sizes,
        budgets=budgets,
        n=n,
        v_inf=v_inf,
        max_order=max_order,
    )
    print(f"[done] wrote {out_path}")

    if n_seeds >= 2:
        print("\n[sanity] empirical Var ~ 1/budget slope per |S|:")
        var = phi.var(axis=0, ddof=1)
        log_b = np.log(budgets.astype(float))
        for k in range(1, max_order + 1):
            mask = sizes == k
            if not mask.any():
                continue
            mean_var = var[:, mask].mean(axis=1)
            slope, _ = np.polyfit(log_b, np.log(mean_var), 1)
            print(f"  |S|={k}: log-log slope = {slope:+.3f}  (expect ~ -1)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--x_index", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "cache" / "variance_bound_msr_independent60.npz",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(out_path=args.out, n_seeds=args.seeds, max_order=args.order, x_index=args.x_index)
