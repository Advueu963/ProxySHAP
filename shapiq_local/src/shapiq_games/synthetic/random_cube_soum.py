"""Synthetic game: sum of independent random sub-cube atoms.

Each atom occupies a random size-``atom_size`` subset of features and assigns an
arbitrary real value to each of the ``2**atom_size`` binary configurations on that
subset. The Möbius transform of each atom is supported entirely on subsets of its
own support, so the global Möbius support is at most
``n_basis_games * 2**atom_size`` regardless of ``n_players``. This makes exact
ground-truth interaction values tractable for very large player counts (n > 40),
provided ``atom_size`` is kept modest (e.g. ``<= 8``).

Compared to ``SOUM`` (which uses unanimity indicators and is therefore
tree-friendly), the random sub-cube atoms are generic functions on each atom's
local cube — no functional form (tree, linear, sparse-Fourier) has a structural
advantage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from shapiq.game_theory.moebius_converter import MoebiusConverter


def _yates_moebius(values: np.ndarray) -> np.ndarray:
    """In-place Möbius transform on a length-``2**t`` lookup via Yates' algorithm.

    Index ``p`` encodes the binary subset (bit ``i`` set = element ``i`` present).
    Returns a copy; runs in O(t * 2**t).
    """
    m = values.astype(np.float64, copy=True)
    size = m.shape[0]
    t = int(round(np.log2(size)))
    if 1 << t != size:
        msg = f"values length {size} is not a power of two"
        raise ValueError(msg)
    for i in range(t):
        mask = 1 << i
        for p in range(size):
            if p & mask:
                m[p] -= m[p ^ mask]
    return m


class RandomCubeSOUM(Game):
    r"""Sum of independent random-valued sub-cube atoms.

    Game value at a coalition ``S`` is

    .. math:: v(S) = \sum_{j=1}^{K} f_j(S \cap T_j)

    where each ``T_j \subseteq [n]`` is a random size-``t`` subset and each
    ``f_j : \{0,1\}^{T_j} \to \mathbb{R}`` is a random real-valued lookup with
    iid Gaussian entries.

    Args:
        n: Number of players in the game.
        n_basis_games: Number of atoms ``K``.
        atom_size: Per-atom support size ``t``. Must satisfy ``1 <= t <= n``.
        random_state: Random seed for atom supports and values.
        normalize: Center the game around the empty coalition value.
        verbose: Print info during construction.

    Attributes:
        atom_supports: Tuple of length-``t`` index tuples ``T_j``.
        atom_values: Array of shape ``(n_basis_games, 2**atom_size)`` with the per-atom lookup tables.
    """

    def __init__(
        self,
        n: int,
        n_basis_games: int,
        atom_size: int,
        *,
        random_state: int | None = None,
        normalize: bool = False,
        verbose: bool = False,
    ) -> None:
        if atom_size < 1 or atom_size > n:
            msg = f"atom_size={atom_size} must be in [1, n_players={n}]."
            raise ValueError(msg)
        if n_basis_games < 1:
            msg = f"n_basis_games={n_basis_games} must be >= 1."
            raise ValueError(msg)

        self._rng = np.random.default_rng(random_state)
        self.n_basis_games: int = n_basis_games
        self.atom_size: int = atom_size

        # Sample atom supports: K tuples of length t
        supports: list[tuple[int, ...]] = []
        for _ in range(n_basis_games):
            chosen = self._rng.choice(n, size=atom_size, replace=False)
            supports.append(tuple(int(v) for v in np.sort(chosen)))
        self.atom_supports: tuple[tuple[int, ...], ...] = tuple(supports)

        # Sample atom values: K x 2^t iid standard normal
        self.atom_values: np.ndarray = self._rng.standard_normal(
            size=(n_basis_games, 1 << atom_size)
        )

        # Powers of two used to decode binary patterns during value evaluation
        self._bit_weights: np.ndarray = (1 << np.arange(atom_size)).astype(np.int64)

        # Pre-compute supports as int arrays for fast fancy indexing
        self._supports_idx: list[np.ndarray] = [
            np.asarray(T, dtype=np.int64) for T in self.atom_supports
        ]

        # Compute Möbius transform once
        self._moebius_coefficients: InteractionValues | None = None
        self.converter: MoebiusConverter | None = None

        empty_value = float(self.value_function(np.zeros((1, n), dtype=np.int64))[0])
        super().__init__(
            n_players=n,
            normalize=normalize,
            verbose=verbose,
            normalization_value=empty_value,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate ``v(S)`` for each coalition row.

        Args:
            coalitions: Binary coalition matrix of shape ``(B, n)``.

        Returns:
            Array of length ``B`` with the game values.
        """
        coalitions_int = coalitions.astype(np.int64, copy=False)
        out = np.zeros(coalitions_int.shape[0], dtype=np.float64)
        for j, support in enumerate(self._supports_idx):
            sub = coalitions_int[:, support]  # (B, t)
            idx = sub @ self._bit_weights  # (B,) integer indices in [0, 2**t)
            out += self.atom_values[j, idx]
        return out

    @property
    def moebius_coefficients(self) -> InteractionValues:
        """The (sparse) Möbius transform of the game."""
        if self._moebius_coefficients is None:
            self._moebius_coefficients = self.moebius_transform()
        return self._moebius_coefficients

    def moebius_transform(self) -> InteractionValues:
        """Compute the sparse global Möbius transform by aggregating per-atom transforms."""
        moebius_dict: dict[tuple[int, ...], float] = {}
        for j, support in enumerate(self.atom_supports):
            local_moebius = _yates_moebius(self.atom_values[j])
            t = self.atom_size
            for p in range(1 << t):
                coef = float(local_moebius[p])
                if coef == 0.0:
                    continue
                global_subset = tuple(support[i] for i in range(t) if (p >> i) & 1)
                moebius_dict[global_subset] = moebius_dict.get(global_subset, 0.0) + coef

        # Drop numerical zeros that emerged from cancellations
        moebius_dict = {k: v for k, v in moebius_dict.items() if abs(v) > 1e-12}

        if not moebius_dict:
            moebius_dict[()] = 0.0

        values = np.empty(len(moebius_dict), dtype=np.float64)
        lookup: dict[tuple[int, ...], int] = {}
        for i, (key, val) in enumerate(moebius_dict.items()):
            values[i] = val
            lookup[key] = i

        baseline_value = moebius_dict.get((), 0.0)
        max_order = max((len(k) for k in moebius_dict), default=0)

        return InteractionValues(
            values=values,
            index="Moebius",
            max_order=max_order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=lookup,
            estimated=False,
            baseline_value=baseline_value,
        )

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Exact interaction values for any supported (``index``, ``order``)."""
        from shapiq.game_theory.moebius_converter import MoebiusConverter

        if self.converter is None:
            self.converter = MoebiusConverter(self.moebius_coefficients)
        return self.converter(index, order)
