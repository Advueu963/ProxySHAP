"""SHAPIQ subclass that computes MSR estimates at many budgets from one sample batch.

Uses the fact that under i.i.d. with-replacement leverage sampling, the SHAPIQ
SII estimator is the running mean of per-sample contributions:
    phi_S(B) = (1/B) * sum_{i=1..B} g_centered(T_i) * w(|S|, |T_i|, |T_i n S|) * P(T_i)^{-1}
with P(T) = 1 / ((n+1) * C(n, |T|)) under leverage sampling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import comb

from shapiq.approximator.montecarlo.shapiq import SHAPIQ

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game


class SHAPIQLeverageTruncated(SHAPIQ):
    """SHAPIQ variant exposing per-budget cumulative estimates from a single sample batch."""

    def _sample_leverage(self, n_samples: int) -> np.ndarray:
        """Sample n_samples coalitions i.i.d. from the leverage distribution."""
        n = self.n
        sizes = self._rng.integers(0, n + 1, size=n_samples)
        coalitions = np.zeros((n_samples, n), dtype=bool)
        for i, s in enumerate(sizes):
            if s == 0:
                continue
            if s == n:
                coalitions[i] = True
            else:
                idx = self._rng.choice(n, size=int(s), replace=False)
                coalitions[i, idx] = True
        return coalitions

    def approximate_at_budgets(
        self,
        budgets: list[int],
        game: Game | Callable[[np.ndarray], np.ndarray],
        chunk_size: int = 2000,
    ) -> dict[int, np.ndarray]:
        """Return SII estimates at each requested budget from one shared sample batch.

        Args:
            budgets: Sorted-or-unsorted list of budgets. Estimates returned for each.
            game: Game callable.
            chunk_size: Interactions per vectorized chunk (memory knob).

        Returns:
            Dict {budget -> values array} where values follow self.interaction_lookup order.
        """
        n = self.n
        max_budget = int(max(budgets))

        coalitions = self._sample_leverage(max_budget)

        empty = np.zeros((1, n), dtype=bool)
        g_empty = float(np.asarray(game(empty))[0])
        g = np.asarray(game(coalitions), dtype=np.float64)
        g_centered = g - g_empty

        T_size = coalitions.sum(axis=1).astype(np.int64)
        p_inv_per_size = (n + 1) * np.array(
            [comb(n, k, exact=True) for k in range(n + 1)], dtype=np.float64
        )
        p_inv_per_T = p_inv_per_size[T_size]
        g_term = (g_centered * p_inv_per_T).astype(np.float64)

        weights = self._get_standard_form_weights("SII")

        n_int = len(self.interaction_lookup)
        budgets_sorted = sorted(set(int(b) for b in budgets))
        out = {b: np.zeros(n_int, dtype=np.float64) for b in budgets_sorted}

        if () in self.interaction_lookup:
            empty_pos = self.interaction_lookup[()]
            for b in budgets_sorted:
                out[b][empty_pos] = g_empty

        by_size: dict[int, list[tuple[tuple[int, ...], int]]] = defaultdict(list)
        for S, pos in self.interaction_lookup.items():
            if 1 <= len(S) <= self.max_order:
                by_size[len(S)].append((S, pos))

        coal_int = coalitions.astype(np.int32)

        for k, items in by_size.items():
            S_list = [S for S, _ in items]
            pos_arr = np.array([pos for _, pos in items], dtype=np.int64)
            m = len(S_list)
            S_mat = np.zeros((m, n), dtype=np.int32)
            for i, S in enumerate(S_list):
                S_mat[i, list(S)] = 1
            w_table = weights[k]

            for start in range(0, m, chunk_size):
                end = min(start + chunk_size, m)
                S_chunk = S_mat[start:end]
                inter_sizes = coal_int @ S_chunk.T
                w_per_sample = w_table[T_size[:, None], inter_sizes]
                contrib = g_term[:, None] * w_per_sample
                csum = np.cumsum(contrib, axis=0)
                pos_chunk = pos_arr[start:end]
                for b in budgets_sorted:
                    est = csum[b - 1] / b
                    out[b][pos_chunk] = est

        return out
