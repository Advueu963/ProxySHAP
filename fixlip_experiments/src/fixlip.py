import time

import shapiq
import numpy as np
import scipy as sp

from . import sampler


class FIxLIP:
    """
    Approximates interaction values using the weighted Banzhaf power index (or Shapley).
    """

    def __init__(
        self,
        n_players=None,
        n_players_image=None,
        n_players_text=None,
        mode="banzhaf",
        p=0.5,
        max_order=2,
        random_state=None,
        sparse_regression=False,
        approximation_type="regression",
    ):
        self.mode = mode
        self.sparse_regression = sparse_regression
        self.is_crossmodal = False
        is_proxyshap = approximation_type in {"proxyshap", "proxyshap-noadjustment"}

        if n_players_image and n_players_text:
            if mode.lower() == "shapley":
                raise ValueError(
                    "approximate_crossmodal() is not available for mode 'Shapley'"
                )
            self.is_crossmodal = True
            n_players = n_players_image + n_players_text
            self.n_players_image = n_players_image
            self.n_players_text = n_players_text

        if mode.lower() == "banzhaf":
            # Sample using uniform weights
            sampling_weights = np.array(
                [
                    sp.special.binom(n_players, k)
                    * (p**k)
                    * ((1 - p) ** (n_players - k))
                    for k in range(n_players + 1)
                ]
            )
            enforce_empty_full = False
        elif mode.lower() == "shapley":
            sampling_weights = np.zeros(n_players + 1)
            # KernelSHAP sampling weights
            for coalition_size in range(1, n_players):
                sampling_weights[coalition_size] = 1 / (
                    coalition_size * (n_players - coalition_size)
            )
            enforce_empty_full = True
        else:
            raise ValueError("`mode` should be either 'Banzhaf' or 'Shapley'.")
        if is_proxyshap:
            enforce_empty_full = True

        if n_players_image and n_players_text:
            self.sampler_image = sampler.CoalitionSampler(
                n_players=n_players_image,
                sampling_weights=np.array(
                    [
                        sp.special.binom(n_players_image, k)
                        * (p**k)
                        * ((1 - p) ** (n_players_image - k))
                        for k in range(n_players_image + 1)
                    ]
                ),
                enforce_empty_full=enforce_empty_full,
                pairing_trick=False,
                random_state=random_state,
            )
            self.sampler_text = sampler.CoalitionSampler(
                n_players=n_players_text,
                sampling_weights=np.array(
                    [
                        sp.special.binom(n_players_text, k)
                        * (p**k)
                        * ((1 - p) ** (n_players_text - k))
                        for k in range(n_players_text + 1)
                    ]
                ),
                enforce_empty_full=enforce_empty_full,
                pairing_trick=False,
                random_state=random_state,
            )
        elif n_players is None:
            raise ValueError(
                "Pass either `n_players` for basic usage or "
                + "pass `n_players_image` and `n_players_text` for crossmodal usage."
            )

        self.n_players = n_players
        self.p = p
        self.max_order = max_order
        self.random_state = random_state
        if approximation_type == "surrogate" and p != 0.5 and mode != "banzhaf":
            raise ValueError(
                "The surrogate works only for FBII. For this use p=0.5 and mode=banzhaf."
            )
        self.approximation_type = approximation_type
        self.sampler = sampler.CoalitionSampler(
            n_players=n_players,
            sampling_weights=sampling_weights,
            enforce_empty_full=enforce_empty_full,
            pairing_trick=False,
            random_state=random_state,
        )
        self.runtime_last_approximate_run = {}  # store runtime of last approximate() call

    def _run_approximator(
        self,
        game,
        coalitions_matrix,
        coalition_values,
        interaction_lookup,
        sampling_adjustment_weights=None,
        cross_modal=False,
    ):
        if self.approximation_type == "regression":
            if cross_modal:  # cross-modal approximation
                kernel_weights_image = np.array(
                    [
                        self.p**k * ((1 - self.p) ** (self.n_players_image - k))
                        for k in range(self.n_players_image + 1)
                    ]
                )
                kernel_weights_text = np.array(
                    [
                        self.p**k * ((1 - self.p) ** (self.n_players_text - k))
                        for k in range(self.n_players_text + 1)
                    ]
                )
                image_regression_weights = get_regression_weights(
                    self.sampler_image, kernel_weights_image
                )
                text_regression_weights = get_regression_weights(
                    self.sampler_text, kernel_weights_text
                )
                regression_weights = np.outer(
                    image_regression_weights, text_regression_weights
                ).reshape(-1)
            else:  # Normal approximation
                if self.mode.lower() == "banzhaf":
                    # set kernel weights for weighted banzhaf
                    kernel_weights = np.array(
                        [
                            self.p**k * ((1 - self.p) ** (self.n_players - k))
                            for k in range(self.n_players + 1)
                        ]
                    )
                elif self.mode.lower() == "shapley":
                    kernel_weights = np.zeros(self.n_players + 1)
                    normalization_constant = 0
                    for coalition_size in range(1, self.n_players):
                        kernel_weights[coalition_size] = 1 / sp.special.binom(
                            self.n_players - 2, coalition_size - 1
                        )
                        normalization_constant += kernel_weights[
                            coalition_size
                        ] * sp.special.binom(self.n_players, coalition_size)
                    # Normalize kernel weights to probability distribution
                    kernel_weights /= normalization_constant
                    big_M = 10e6
                    kernel_weights[0] = big_M
                    kernel_weights[-1] = big_M
                regression_weights = get_regression_weights(
                    self.sampler, kernel_weights
                )
                coalitions_matrix = self.sampler.coalitions_matrix
            # aggregate coalition values
            interaction_values = self.aggregate(
                coalition_matrix=coalitions_matrix,
                regression_weights=regression_weights,
                coalition_values=coalition_values,
                interaction_lookup=interaction_lookup,
            )
        elif self.approximation_type == "proxyshap":
            from proxyshap.proxyshap import ProxySHAP
            from xgboost import XGBRegressor
            if sampling_adjustment_weights is None:
                sampling_adjustment_weights = self.sampler.sampling_adjustment_weights
            xgboost_model = XGBRegressor(
                n_estimators=2000,
                max_depth=3,
                learning_rate=0.05,
                reg_lambda=5,
                random_state=self.random_state,
            )

            approximator = ProxySHAP(
                n=self.n_players,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                random_state=self.random_state,
                proxy_model=xgboost_model,
                adjustment="msr",
                pairing_trick=True,
            )

            (
                coalitions_matrix,
                coalition_values,
                sampling_adjustment_weights,
            ) = self._ensure_empty_full_first(
                coalitions_matrix,
                coalition_values,
                sampling_adjustment_weights,
            )
            interaction_values = approximator.approximate_given(
                coalitions_matrix=coalitions_matrix,
                game_values=coalition_values,
                baseline_value=0.0,
                sampling_adjustment_weights=sampling_adjustment_weights,
            )
        elif self.approximation_type == "proxyspex":
            from shapiq.approximator.sparse.proxyspex import ProxySPEXXGBoost
            from xgboost import XGBRegressor

            xgboost_model = XGBRegressor(
                n_estimators=2000,
                max_depth=3,
                learning_rate=0.05,
                reg_lambda=5,
                random_state=self.random_state,
            )
            approximator = ProxySPEXXGBoost(
                n=self.n_players,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                random_state=self.random_state,
                proxy_model=xgboost_model,
            )
            # ProxySPEX does not need baseline_value
            interaction_values = approximator.approximate_given(
                coalitions_matrix=coalitions_matrix,
                game_values=coalition_values,
            )
        elif self.approximation_type == "proxyshap-noadjustment":
            from proxyshap.proxyshap import ProxySHAP
            from xgboost import XGBRegressor

            xgboost_model = XGBRegressor(
                n_estimators=2000,
                max_depth=3,
                learning_rate=0.05,
                reg_lambda=5,
                random_state=self.random_state,
            )

            approximator = ProxySHAP(
                n=self.n_players,
                index="FBII" if self.mode.lower() == "banzhaf" else "FSII",
                max_order=self.max_order,
                random_state=self.random_state,
                proxy_model=xgboost_model,
                adjustment="none",
                pairing_trick=True,
            )

            if sampling_adjustment_weights is None:
                sampling_adjustment_weights = self.sampler.sampling_adjustment_weights
            (
                coalitions_matrix,
                coalition_values,
                sampling_adjustment_weights,
            ) = self._ensure_empty_full_first(
                coalitions_matrix,
                coalition_values,
                sampling_adjustment_weights,
            )
            interaction_values = approximator.approximate_given(
                coalitions_matrix=coalitions_matrix,
                game_values=coalition_values,
                baseline_value=0.0,
            )
        else:
            raise ValueError(
                f"`approximation_type` should be either 'regression', 'proxyshap', 'proxyspex' or 'proxyshap-noadjustment'. Got `{self.approximation_type}`."
            )
        return interaction_values

    def approximate(self, game, budget, interaction_lookup=None, time_game=False):
        approximation_start_time = time.time()
        # sample coalitions
        self.sampler.sample(budget)
        # evaluate coalition values (un-normalized game call)
        coalition_values = game.value_function(self.sampler.coalitions_matrix)
        self.time_game_end = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            self.time_game_end - approximation_start_time
        )
        coalition_values = coalition_values - game.normalization_value
        # run approximator
        interaction_values = self._run_approximator(
            game, self.sampler.coalitions_matrix, coalition_values, interaction_lookup
        )

        end_time = time.time()
        self.runtime_last_approximate_run["total"] = end_time - approximation_start_time
        if self.max_order == 1:
            # Ensure that all player values are present for first-order interactions.
            # Pad with zeros if necessary.
            pad_interactions = {(i,): 0 for i in range(self.n_players)}
            pad_interactionvalues = shapiq.InteractionValues(
                values=pad_interactions,
                baseline_value=0,
                n_players=self.n_players,
                index=interaction_values.index,
                max_order=1,
                min_order=1,
                estimated=interaction_values.estimated,
                estimation_budget=interaction_values.estimation_budget,
            )
            interaction_values += pad_interactionvalues
        if self.max_order == 2:
            pad_interactions = {
                (i, j): 0
                for i in range(self.n_players)
                for j in range(i, self.n_players)
            }
            pad_interactions.update({(i,): 0 for i in range(self.n_players)})
            pad_interactionvalues = shapiq.InteractionValues(
                values=pad_interactions,
                baseline_value=0,
                n_players=self.n_players,
                index=interaction_values.index,
                max_order=2,
                min_order=1,
                estimated=interaction_values.estimated,
                estimation_budget=interaction_values.estimation_budget,
            )
            interaction_values += pad_interactionvalues
        return interaction_values

    def approximate_crossmodal(
        self,
        game,
        budget=None,
        budget_image=None,
        budget_text=None,
        interaction_lookup=None,
    ):
        approximation_start_time = time.time()
        if not self.is_crossmodal:
            raise ValueError(
                "Crossmodal approximation is not initialized."
                + "Pass `n_players_image` and `n_players_text` to FIxLIP()."
            )
        # split budget based on n_players_text and n_players_image
        if budget is not None:
            if budget < 4:
                raise ValueError("`budget` should be at least 4.")
            budget_image, budget_text = self.split_budget(budget)
        elif budget_image is None or budget_text is None:
            raise ValueError(
                "Pass either `budget` or `budget_image` and `budget_text`."
            )
        # sample coalitions from both modalities
        # print(
        #     "BUDGET", budget, "budget_image", budget_image, "budget_text", budget_text
        # )
        self.sampler_image.sample(budget_image)
        self.sampler_text.sample(budget_text)
        # evaluate coalition values efficiently with _crossmodal (un-normalized game call)
        # print(
        #     "Sampled coalitions image shape: ",
        #     self.sampler_image.coalitions_matrix.shape,
        # )
        # print(
        #     "Sampled coalitions text shape: ", self.sampler_text.coalitions_matrix.shape
        # )
        coalition_values_crossmodal = game.value_function_crossmodal(
            coalitions_image=self.sampler_image.coalitions_matrix,
            coalitions_text=self.sampler_text.coalitions_matrix,
        )
        self.time_game_end = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            self.time_game_end - approximation_start_time
        )
        coalition_values_crossmodal = (
            coalition_values_crossmodal - game.normalization_value
        )
        # reshape inputs to aggregate()
        coalition_values = coalition_values_crossmodal.reshape(-1)
        n_image = self.sampler_image.coalitions_matrix.shape[0]
        n_text = self.sampler_text.coalitions_matrix.shape[0]
        coalitions_matrix = np.concatenate(
            [
                np.repeat(self.sampler_image.coalitions_matrix, n_text, axis=0),
                np.tile(self.sampler_text.coalitions_matrix, (n_image, 1)),
            ],
            axis=1,
        )

        sampling_adjustment_weights = None
        if self.approximation_type in {"proxyshap", "proxyshap-noadjustment"}:
            sampling_adjustment_weights = np.outer(
                self.sampler_image.sampling_adjustment_weights,
                self.sampler_text.sampling_adjustment_weights,
            ).reshape(-1)
        interaction_values = self._run_approximator(
            game,
            coalitions_matrix,
            coalition_values,
            interaction_lookup,
            sampling_adjustment_weights=sampling_adjustment_weights,
            cross_modal=True,
        )

        end_time = time.time()
        self.runtime_last_approximate_run["total"] = end_time - approximation_start_time

        if self.max_order == 1:
            # Ensure that all player values are present for first-order interactions.
            # Pad with zeros if necessary.
            pad_interactions = {(i,): 0 for i in range(self.n_players)}
            pad_interactionvalues = shapiq.InteractionValues(
                values=pad_interactions,
                baseline_value=0,
                n_players=self.n_players,
                index=interaction_values.index,
                max_order=1,
                min_order=1,
                estimated=interaction_values.estimated,
                estimation_budget=interaction_values.estimation_budget,
            )
            interaction_values += pad_interactionvalues
        if self.max_order == 2:
            pad_interactions = {
                (i, j): 0
                for i in range(self.n_players)
                for j in range(i, self.n_players)
            }
            pad_interactions.update({(i,): 0 for i in range(self.n_players)})
            pad_interactionvalues = shapiq.InteractionValues(
                values=pad_interactions,
                baseline_value=0,
                n_players=self.n_players,
                index=interaction_values.index,
                max_order=2,
                min_order=1,
                estimated=interaction_values.estimated,
                estimation_budget=interaction_values.estimation_budget,
            )
            interaction_values += pad_interactionvalues
        return interaction_values

    def _ensure_empty_full_first(
        self,
        coalitions_matrix: np.ndarray,
        coalition_values: np.ndarray,
        sampling_adjustment_weights: np.ndarray | None,
    ):
        coalition_sizes = np.sum(coalitions_matrix, axis=1)
        empty_indices = np.where(coalition_sizes == 0)[0]
        full_indices = np.where(coalition_sizes == coalitions_matrix.shape[1])[0]
        if empty_indices.size == 0:
            raise ValueError("Empty coalition missing from sampled coalitions.")
        empty_idx = int(empty_indices[0])
        full_idx = int(full_indices[0]) if full_indices.size > 0 else None
        ordered = [empty_idx]
        if full_idx is not None and full_idx != empty_idx:
            ordered.append(full_idx)
        ordered.extend(i for i in range(coalitions_matrix.shape[0]) if i not in ordered)
        coalitions_matrix = coalitions_matrix[ordered]
        coalition_values = coalition_values[ordered]
        if sampling_adjustment_weights is not None:
            sampling_adjustment_weights = sampling_adjustment_weights[ordered]
        return coalitions_matrix, coalition_values, sampling_adjustment_weights

    def aggregate(
        self,
        coalition_matrix,
        regression_weights,
        coalition_values,
        interaction_lookup: dict | None = None,
    ) -> shapiq.InteractionValues:
        """Aggregates the coalition values using the weighted Banzhaf power index."""
        n_coalitions, n_players = np.shape(coalition_matrix)
        # populate interactions to use for regression
        if interaction_lookup is None:  # first check if interaction_lookup is passed
            interaction_lookup = shapiq.utils.generate_interaction_lookup(
                set(range(n_players)), min_order=0, max_order=self.max_order
            )
        n_interactions = len(interaction_lookup)
        # set response, subtract baseline for better approximation, it will be added later
        regression_response = coalition_values.copy()
        # create regression matrix
        regression_matrix = np.zeros((n_coalitions, n_interactions))
        for i, interaction in enumerate(interaction_lookup.keys()):
            regression_matrix[:, i] = coalition_matrix[:, interaction].prod(axis=1)
        # solve regression
        values = solve_regression(
            regression_matrix,
            regression_response,
            regression_weights,
            sparse_regression=self.sparse_regression,
        )
        # return interaction values
        interaction_values = shapiq.InteractionValues(
            values=values,
            interaction_lookup=interaction_lookup,
            baseline_value=values[interaction_lookup[()]],
            n_players=n_players,
            index="Moebius",
            max_order=self.max_order,
            min_order=0,
            estimated=False if n_coalitions >= 2**n_players else True,
            estimation_budget=n_coalitions,
        )
        interaction_values.index = "FSII" if self.mode.lower() == "shapley" else "FWBII"
        return interaction_values

    #:# ---------- utility functions ---------- #:#

    def split_budget(self, budget):
        """
        Heuristic to choose a reasonable budget split.
        """
        # print(self.n_players_text, self.n_players_image)
        if self.n_players_text < self.n_players_image:
            budget_text = np.sqrt(budget) * self.n_players_text / self.n_players_image
            budget_text = int(np.ceil(np.max([4, budget_text])))
            budget_text = int(np.min([2**self.n_players_text, budget_text]))
            budget_image = int(budget / budget_text)
        else:
            budget_image = np.sqrt(budget) * self.n_players_image / self.n_players_text
            budget_image = int(np.ceil(np.max([4, budget_image])))
            budget_image = int(np.min([2**self.n_players_image, budget_image]))
            budget_text = int(budget / budget_image)
        return budget_image, budget_text


def solve_regression(
    X: np.ndarray, y: np.ndarray, kernel_weights: np.ndarray, sparse_regression=False
) -> np.ndarray:
    if not sparse_regression:
        try:
            # try solving via solve function
            WX = kernel_weights[:, np.newaxis] * X
            phi = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            # solve WLSQ via lstsq function and throw warning
            W_sqrt = np.sqrt(kernel_weights)
            X = W_sqrt[:, np.newaxis] * X
            y = W_sqrt * y
            phi = np.linalg.lstsq(X, y, rcond=None)[0]
    else:
        X_sparse = sp.sparse.csr_matrix(X)
        W_sparse = sp.sparse.diags(kernel_weights)
        WX_sparse = W_sparse @ X_sparse  # Pre-multiply: W * X and W * y
        Wy = kernel_weights * y
        result = sp.sparse.linalg.lsqr(WX_sparse, Wy)  # Solve sparse least squares
        phi = result[0]  # coefficients
        return phi
    return phi


def get_regression_weights(sampler, kernel_weights):
    """Computes the regression weights, requires that sampling weights are proportional to kernel weights.
    Regression weights are equal to kernel weights for coalitions that are not sampled (using the border-trick).
    Otherwise, the regression weights are set to the empirical averages, i.e. # occurrences / # sampled coalitions.
    """
    regression_weights_not_sampled = kernel_weights[
        np.sum(sampler.coalitions_matrix, axis=1)
    ]
    regression_weights = sampler.empirical_occurrences
    regression_weights[~sampler.is_coalition_sampled] = (
        regression_weights[~sampler.is_coalition_sampled]
        * regression_weights_not_sampled[~sampler.is_coalition_sampled]
    )
    return regression_weights
