"""Base Sparse approximator for fourier-based interaction computation."""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict
import time
from typing import TYPE_CHECKING, Literal, cast, get_args

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from sparse_transform.qsft.qsft import transform as sparse_fourier_transform
from sparse_transform.qsft.signals.input_signal_subsampled import (
    SubsampledSignal as SubsampledSignalFourier,
)
from sparse_transform.qsft.utils.general import fourier_to_mobius as fourier_to_moebius
from sparse_transform.qsft.utils.query import get_bch_decoder

from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.game_theory.moebius_converter import (
    MoebiusConverter,
    ValidMoebiusConverterIndices,
)
from shapiq.interaction_values import InteractionValues
import xgboost as xgb

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from shapiq.game import Game

ValidSparseIndices = ValidMoebiusConverterIndices


class Sparse(Approximator[ValidSparseIndices]):
    """Approximator interface using sparse transformation techniques.

    This class implements a sparse approximation method for computing various interaction indices
    using sparse Fourier transforms. It efficiently estimates interaction values with a limited
    sample budget by leveraging sparsity in the Fourier domain. The notion of sparse approximation
    is described in [Kan25]_ and further improved in [But25]_.

    See Also:
        - :class:`~shapiq.approximator.sparse.SPEX` for a specific implementation of the
            sparse approximation using Fourier transforms described in [Kan25]_.
        - :class:`~shapiq.approximator.sparse.ProxySPEX` for a specific implementation of the
            sparse approximation using Fourier transforms described in [But25]_.

    Attributes:
        transform_type: Type of transform used (currently only ``"fourier"`` is supported).

        degree_parameter: A parameter that controls the maximum degree of the interactions to
            extract during execution of the algorithm. Note that this is a soft limit, and in
            practice, the algorithm may extract interactions of any degree. We typically find
            that there is little value going beyond ``5``. Defaults to ``5``. Note that
            increasing this parameter will need more ``budget`` in the :meth:`approximate`
            method.

        query_args: Parameters for querying the signal.

        decoder_args: Parameters for decoding the transform.

    Raises:
        ValueError: If transform_type is not "fourier" or if decoder_type is not "soft" or "hard".

    References:
        .. [Kan25] Kang, J.S., Butler, L., Agarwal. A., Erginbas, Y.E., Pedarsani, R., Ramchandran, K., Yu, Bin (2025). SPEX: Scaling Feature Interaction Explanations for LLMs https://arxiv.org/abs/2502.13870
        .. [But25] Butler, L., Kang, J.S., Agarwal. A., Erginbas, Y.E., Yu, Bin, Ramchandran, K. (2025). ProxySPEX: Inference-Efficient Interpretability via Sparse Feature Interactions in LLMs https://arxiv.org/pdf/2505.17495
    """

    valid_indices: tuple[ValidSparseIndices, ...] = tuple(get_args(ValidSparseIndices))  # type: ignore[assignment]
    """The valid indices for the SPEX approximator."""

    def __init__(
        self,
        n: int,
        index: ValidSparseIndices,
        *,
        max_order: int | None = None,
        top_order: bool = False,
        random_state: int | None = None,
        transform_type: Literal["fourier"] = "fourier",
        decoder_type: Literal[
            "soft",
            "hard",
            "proxyspex",
            "proxyspex_notrunjaction",
            "proxyspex_.7trunjaction",
            "proxyspex_xgboost",
            "proxyspex_noadjustment",
            "proxyspex_xgboost_notrunjaction",
            "proxyspex_notrunjaction_norefinement",
            "proxyspex_xgboost_notrunjaction_norefinement"
        ]
        | None = "proxyspex",
        degree_parameter: int = 5,
        proxy_model: Any = None,
        cut_off_quantile: float = 0.95,
    ) -> None:
        """Initialize the Sparse approximator.

        Args:
            n: Number of players (features).

            max_order: Maximum interaction order to consider. Defaults to ``None``, which means
                that all orders up to ``n`` will be considered.

            index: The Interaction index to use. All indices supported by shapiq's
                :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter` are supported.

            top_order: If ``True``, only reports interactions of exactly order ``max_order``.
                Otherwise, reports all interactions up to order ``max_order``. Defaults to
                ``False``.

            random_state: Seed for random number generator. Defaults to ``None``.

            transform_type: Type of transform to use. Currently only "fourier" is supported.

            decoder_type: Type of decoder to use, either "soft", "hard", or "proxyspex". Defaults to "proxyspex".

            degree_parameter: A parameter that controls the maximum degree of the interactions to
                extract during execution of the algorithm. Note that this is a soft limit, and in
                practice, the algorithm may extract interactions of any degree. We typically find
                that there is little value going beyond ``5``. Defaults to ``5``. Note that
                increasing this parameter will need more ``budget`` in the :meth:`approximate`
                method.

        """
        if transform_type.lower() not in ["fourier"]:
            msg = "transform_type must be 'fourier'"
            raise ValueError(msg)
        self.transform_type = transform_type.lower()
        self.degree_parameter = degree_parameter
        max_order = n if max_order is None else max_order
        self.decoder_type = "proxyspex" if decoder_type is None else decoder_type.lower()
        if self.decoder_type not in [
            "soft",
            "hard",
            "proxyspex",
            "proxyspex_notrunjaction",
            "proxyspex_.7trunjaction",
            "proxyspex_xgboost",
            "proxyspex_noadjustment",
            "proxyspex_xgboost_notrunjaction",
            "proxyspex_xgboost_notrunjaction_norefinement",
            "proxyspex_xgboost_norefinement",
            "proxyspex_notrunjaction_norefinement",
            "proxyspex_nogridsearch"
        ]:
            msg = "decoder_type must be 'soft', 'hard', 'proxyspex', 'proxyspex_notrunjaction', 'proxyspex_.7trunjaction', 'proxyspex_xgboost', or 'proxyspex_noadjustment'"
            raise ValueError(msg)
        if self.decoder_type == "proxyspex":
            try:
                import lightgbm as lgb  # noqa: F401
            except ImportError as err:
                msg = (
                    "The 'lightgbm' package is required when decoder_type is 'proxyspex' but it is "
                    "not installed. Please see the installation instructions at "
                    "https://github.com/microsoft/LightGBM/tree/master/python-package."
                )
                raise ImportError(msg) from err
        # The sampling parameters for the Fourier transform
        self.query_args = {
            "query_method": "complex",
            "num_subsample": 3,
            "delays_method_source": "joint-coded",
            "subsampling_method": "qsft",
            "delays_method_channel": "identity-siso",
            "num_repeat": 1,
            "t": self.degree_parameter,
        }
        if self.decoder_type.startswith("proxyspex"):
            self.decoder_args = {
                "max_depth": [3, 5],
                "max_iter": [500, 1000],
                "learning_rate": [0.01, 0.1],
            }
            # Apply log-sum-exp trick for numerical stability, then exponentiate to get weights
            # binomial coeffiecient = gamma(n+1) / (gamma(k+1) * gamma(n-k+1))
            log_weights = np.array(
                [
                    math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    for i in range(n + 1)
                ],
                dtype=np.float64,
            )
            scaled_weights = np.exp(log_weights - np.max(log_weights))
            tiny = np.finfo(np.float64).tiny
            scaled_weights = np.clip(scaled_weights, tiny, None)
            self._uniform_sampler = CoalitionSampler(
                n_players=n,
                sampling_weights=np.ones(n + 1),
                pairing_trick=True,
                replacement=False,
                random_state=random_state,
            )
        else:
            self.decoder_args = {
                "num_subsample": 3,
                "num_repeat": 1,
                "reconstruct_method_source": "coded",
                "peeling_method": "multi-detect",
                "reconstruct_method_channel": (
                    "identity-siso" if self.decoder_type == "soft" else "identity"
                ),
                "regress": "lasso",
                "res_energy_cutoff": 0.9,
                "source_decoder": get_bch_decoder(n, self.degree_parameter, self.decoder_type),
            }
        self.cut_off_quantile = cut_off_quantile
        self.proxy_model = proxy_model
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            top_order=top_order,
            random_state=random_state,
            initialize_dict=False,  # Important for performance
        )

    def approximate_given(
        self,
        coalitions_matrix: np.ndarray,
        game_values: np.ndarray,
        **kwargs: Any,  # noqa: ARG002
    ):
        budget = coalitions_matrix.shape[0]
        approximation_start_time = time.time()
        if self.decoder_type.startswith("proxyspex"):
            import lightgbm as lgb

            used_budget = budget

            # Take the budget amount of uniform samples
            sample_end_time = time.time()
            self.runtime_last_approximate_run["sampling_time"] = (
                sample_end_time - approximation_start_time
            )
            train_X = pd.DataFrame(
                coalitions_matrix,
                columns=np.array([f"f{i}" for i in range(self.n)]),
            )
            train_y = game_values
            game_evaluation_end_time = time.time()
            self.runtime_last_approximate_run["evaluations"] = (
                game_evaluation_end_time - sample_end_time
            )
            if not self.decoder_type.startswith("proxyspex_xgboost"):
                base_model = lgb.LGBMRegressor(
                    verbose=-1, n_jobs=1, random_state=self._random_state
                )

                # Set up GridSearchCV with cross-validation
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=self.decoder_args,
                    scoring="r2",
                    cv=5,
                    verbose=0,
                    n_jobs=1,
                )

                # Fit the model on the training data
                grid_search.fit(train_X, train_y)

                best_model = grid_search.best_estimator_
                model_training_end_time = time.time()
                self.runtime_last_approximate_run["model_training_time"] = (
                    model_training_end_time - game_evaluation_end_time
                )
                initial_transform = self._refine(
                    self._lgboost_to_fourier(best_model.booster_.dump_model()),
                    coalitions_matrix,
                    train_y,
                )
                refine_end_time = time.time()
                self.runtime_last_approximate_run["refinement_time"] = (
                    refine_end_time - model_training_end_time
                )
            elif self.decoder_type.startswith("proxyspex_xgboost"):
                base_model = xgb.XGBRegressor(
                    n_jobs=-1,
                    tree_method="hist",
                    objective="reg:squarederror",
                    random_state=self._random_state,
                ) if self.proxy_model is None else self.proxy_model
                base_model.fit(train_X, train_y)
                model_training_end_time = time.time()
                self.runtime_last_approximate_run["model_training_time"] = (
                    model_training_end_time - game_evaluation_end_time
                )
                initial_transform = self._refine(
                    self._xgboost_to_fourier(base_model.get_booster().get_dump(dump_format="json")),
                    coalitions_matrix,
                    train_y,
                )
                refine_end_time = time.time()
                self.runtime_last_approximate_run["refinement_time"] = (
                    refine_end_time - model_training_end_time
                )
        else:
            raise NotImplementedError("Spex is not coverd for the given coalitions_matrix.")
            ## Not Implemented for given coalitions_matrix yet
            # # Find the max value of b that fits within the given sample budget and get the used budget
            # used_budget = self._set_transform_budget(budget)
            # signal = SubsampledSignalFourier(
            #     func=lambda inputs: game(inputs.astype(bool)),
            #     n=self.n,
            #     q=2,
            #     query_args=self.query_args,
            # )
            # # Extract the coefficients of the original transform
            # initial_transform = {
            #     tuple(np.nonzero(key)[0]): np.real(value)
            #     for key, value in sparse_fourier_transform(
            #         signal, **self.decoder_args
            #     ).items()
            # }
            # model_training_end_time = time.time()
            # self.runtime_last_approximate_run["model_training_time"] = (
            #     model_training_end_time - approximation_start_time
            # )
        # If we are using the fourier transform, we need to convert it to a Moebius transform
        moebius_transform = fourier_to_moebius(initial_transform)
        transform_end_time = time.time()
        self.runtime_last_approximate_run["fourier_transform_time"] = transform_end_time - (
            refine_end_time if self.decoder_type == "proxyspex" else model_training_end_time
        )
        # Convert the Moebius transform to the desired index
        result = self._process_moebius(moebius_transform=moebius_transform)
        finalize_end_time = time.time()
        self.runtime_last_approximate_run["process_moebius_time"] = (
            finalize_end_time - transform_end_time
        )
        # Filter the output as needed
        if self.top_order:
            result = self._filter_order(result)
        # finalize the interactions
        interaction = InteractionValues(
            values=result,
            index=self.approximation_index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            estimated=True,
            estimation_budget=used_budget,
            baseline_value=(
                result[self.interaction_lookup[()]] if () in self.interaction_lookup else 0.0
            ),
            target_index=self.index,
        )
        total_end_time = time.time()
        self.runtime_last_approximate_run["total"] = total_end_time - approximation_start_time
        return interaction

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximates the interaction values using a sparse transform approach.

        Args:
            budget: The budget for the approximation.
            game: The game function that returns the values for the coalitions.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            The approximated Shapley interaction values.
        """
        approximation_start_time = time.time()
        if self.decoder_type.startswith("proxyspex"):
            import lightgbm as lgb

            used_budget = budget

            # Take the budget amount of uniform samples
            self._uniform_sampler.sample(budget)
            sample_end_time = time.time()
            self.runtime_last_approximate_run["sampling_time"] = (
                sample_end_time - approximation_start_time
            )
            train_X = pd.DataFrame(
                self._uniform_sampler.coalitions_matrix,
                columns=np.array([f"f{i}" for i in range(self.n)]),
            )
            train_y = game(self._uniform_sampler.coalitions_matrix)
            game_evaluation_end_time = time.time()
            self.runtime_last_approximate_run["evaluations"] = (
                game_evaluation_end_time - sample_end_time
            )
            if not self.decoder_type.startswith("proxyspex_xgboost"):
                base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=self._random_state)
                if self.decoder_type.endswith("_nogridsearch"):
                    base_model.fit(train_X, train_y)
                    best_model = base_model
                else:
                    # Set up GridSearchCV with cross-validation
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=self.decoder_args,
                        scoring="r2",
                        cv=5,
                        verbose=0,
                        n_jobs=1,
                    )

                    # Fit the model on the training data
                    grid_search.fit(train_X, train_y)

                    best_model = grid_search.best_estimator_
                model_training_end_time = time.time()
                self.runtime_last_approximate_run["model_training_time"] = (
                    model_training_end_time - game_evaluation_end_time
                )
                initial_transform = self._refine(
                        self._lgboost_to_fourier(best_model.booster_.dump_model()),
                        self._uniform_sampler.coalitions_matrix,
                        train_y,
                )
                refine_end_time = time.time()
                self.runtime_last_approximate_run["refinement_time"] = (
                    refine_end_time - model_training_end_time
                )
            elif self.decoder_type.startswith("proxyspex_xgboost"):
                base_model = xgb.XGBRegressor(
                    n_jobs=-1,
                    tree_method="hist",
                    objective="reg:squarederror",
                    random_state=self._random_state,
                ) if self.proxy_model is None else self.proxy_model
                base_model.fit(train_X, train_y)
                model_training_end_time = time.time()
                self.runtime_last_approximate_run["model_training_time"] = (
                    model_training_end_time - game_evaluation_end_time
                )
                initial_transform = self._refine(
                        self._xgboost_to_fourier(base_model.get_booster().get_dump(dump_format="json")),
                        self._uniform_sampler.coalitions_matrix,
                        train_y,
                )
                refine_end_time = time.time()
                self.runtime_last_approximate_run["refinement_time"] = (
                    refine_end_time - model_training_end_time
                )
        else:
            # Find the max value of b that fits within the given sample budget and get the used budget
            used_budget = self._set_transform_budget(budget)
            signal = SubsampledSignalFourier(
                func=lambda inputs: game(inputs.astype(bool)),
                n=self.n,
                q=2,
                query_args=self.query_args,
            )
            # Extract the coefficients of the original transform
            initial_transform = {
                tuple(np.nonzero(key)[0]): np.real(value)
                for key, value in sparse_fourier_transform(signal, **self.decoder_args).items()
            }
            model_training_end_time = time.time()
            self.runtime_last_approximate_run["model_training_time"] = (
                model_training_end_time - approximation_start_time
            )
        # If we are using the fourier transform, we need to convert it to a Moebius transform
        print(f"Initial Fourier coefficients extracted: {len(initial_transform)}")
        moebius_transform = fourier_to_moebius(initial_transform)
        transform_end_time = time.time()
        print(f"Moebius coefficients after conversion: {len(moebius_transform)}")
        self.runtime_last_approximate_run["fourier_transform_time"] = transform_end_time - (
            refine_end_time if self.decoder_type.startswith("proxyspex") else model_training_end_time
        )
        # Convert the Moebius transform to the desired index

        result = self._process_moebius(moebius_transform=moebius_transform)
        print(f"Final interaction values computed: {len(result)}")

        finalize_end_time = time.time()
        self.runtime_last_approximate_run["process_moebius_time"] = (
            finalize_end_time - transform_end_time
        )
        # Filter the output as needed
        if self.top_order:
            result = self._filter_order(result)
        # finalize the interactions
        interaction = InteractionValues(
            values=result,
            index=self.approximation_index,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup=copy.deepcopy(self.interaction_lookup),
            estimated=True,
            estimation_budget=used_budget,
            baseline_value=(
                result[self.interaction_lookup[()]] if () in self.interaction_lookup else 0.0
            ),
            target_index=self.index,
        )
        total_end_time = time.time()
        self.runtime_last_approximate_run["total"] = total_end_time - approximation_start_time
        return interaction

    def _filter_order(self, result: np.ndarray) -> np.ndarray:
        """Filters the interactions to keep only those of the maximum order.

        This method is used when top_order=True to filter out all interactions that are not
        of exactly the maximum order (self.max_order).

        Args:
            result: Array of interaction values.

        Returns:
            Filtered array containing only interaction values of the maximum order.
            The method also updates the internal _interaction_lookup dictionary.
        """
        filtered_interactions = {}
        filtered_results = []
        i = 0
        for j, key in enumerate(self.interaction_lookup):
            if len(key) == self.max_order:
                filtered_interactions[key] = i
                filtered_results.append(result[j])
                i += 1
        self._interaction_lookup = filtered_interactions
        return np.array(filtered_results)

    def _process_moebius(self, moebius_transform: dict[tuple, float]) -> np.ndarray:
        """Convert the Moebius transform into the desired index.

        Args:
            moebius_transform: The Moebius transform to process as a dict mapping tuples to float
                values.

        Returns:
            np.ndarray: The converted interaction values based on the specified index.
            The function also updates the internal _interaction_lookup dictionary.
        """
        moebius_interactions = InteractionValues(
            values=np.array([moebius_transform[key] for key in moebius_transform]),
            index="Moebius",
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            interaction_lookup={key: i for i, key in enumerate(moebius_transform.keys())},
            estimated=True,
            baseline_value=moebius_transform.get((), 0.0),
        )
        autoconverter = MoebiusConverter(moebius_coefficients=moebius_interactions)
        converted_interaction_values = autoconverter(
            index=cast(ValidMoebiusConverterIndices, self.index), order=self.max_order
        )
        self._interaction_lookup = converted_interaction_values.interaction_lookup
        return converted_interaction_values.values  # noqa: PD011

    def _set_transform_budget(self, budget: int) -> int:
        """Sets the appropriate transform budget parameters based on the given sample budget.

        This method calculates the maximum possible 'b' parameter (number of bits to subsample)
        that fits within the provided budget, then configures the query and decoder arguments
        accordingly. The actual number of samples that will be used is returned.

        Args:
            budget: The maximum number of samples allowed for the approximation.

        Returns:
            int: The actual number of samples that will be used, which is less than or equal to the
                budget.

        Raises:
            ValueError: If the budget is too low to compute the transform with acceptable parameters.
        """
        b = SubsampledSignalFourier.get_b_for_sample_budget(
            budget, self.n, self.degree_parameter, 2, self.query_args
        )
        used_budget = SubsampledSignalFourier.get_number_of_samples(
            self.n, b, self.degree_parameter, 2, self.query_args
        )

        if b <= 2:
            while self.degree_parameter > 2:
                self.degree_parameter -= 1
                self.query_args["t"] = self.degree_parameter

                # Recalculate 'b' with the updated 't'
                b = SubsampledSignalFourier.get_b_for_sample_budget(
                    budget, self.n, self.degree_parameter, 2, self.query_args
                )

                # Compute the used budget
                used_budget = SubsampledSignalFourier.get_number_of_samples(
                    self.n, b, self.degree_parameter, 2, self.query_args
                )

                # Break if 'b' is now sufficient
                if b > 2:
                    self.decoder_args["source_decoder"] = get_bch_decoder(
                        self.n, self.degree_parameter, self.decoder_type
                    )
                    break

            # If 'b' is still too low, raise an error
            if b <= 2:
                msg = (
                    "Insufficient budget to compute the transform. Increase the budget or use a "
                    "different approximator."
                )
                raise ValueError(msg)
        # Store the final 'b' value
        self.query_args["b"] = b
        self.decoder_args["b"] = b
        return used_budget

    def _lgboost_to_fourier(self, model_dict: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Extracts the aggregated Fourier coefficients from an LGBoost model dictionary.

        This method iterates over all trees in the LightGBM ensemble, computes the
        Fourier coefficients for each individual tree using the `_lgboost_tree_to_fourier`
        helper method, and then sums these coefficients to get the final Fourier
        representation of the complete model.

        Args:
        model_dict: A dictionary representing the trained LGBoost model, as
            produced by `model.booster_.dump_model()`.

        Returns:
            A dictionary that maps interaction tuples (representing Fourier frequencies)
            to their aggregated Fourier coefficients.
        """
        aggregated_coeffs = defaultdict(float)

        for tree_info in model_dict["tree_info"]:
            tree_coeffs = self._lgboost_tree_to_fourier(tree_info)
            for interaction, value in tree_coeffs.items():
                aggregated_coeffs[interaction] += value

        # Convert defaultdict to a standard dict, removing zero-valued coefficients
        return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

    def _lgboost_tree_to_fourier(self, tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Recursively strips the Fourier coefficients from a single LGBoost tree.

        This method traverses a tree's structure, as provided by LightGBM's `dump_model`
        method, and computes the Fourier representation of the piecewise-constant
        function that the tree defines. The logic is adapted from the work by Gorji et al. (2024).

        Args:
            tree_info: A dictionary representing a single decision tree from an LGBM model.

        Returns:
            A dictionary mapping interaction tuples to their corresponding coefficients for
            the single tree.

        References:
            Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
            "SHAP values via sparse Fourier representation"
            arXiv preprint arXiv:2410.06300 (2024).
        """

        def _combine_coeffs(
            left_coeffs: dict[tuple[int, ...], float],
            right_coeffs: dict[tuple[int, ...], float],
            feature_idx: int,
        ) -> dict[tuple[int, ...], float]:
            """Combines Fourier coefficients from the left and right children of a split node."""
            combined_coeffs = {}
            all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

            for interaction in all_interactions:
                left_val = left_coeffs.get(interaction, 0.0)
                right_val = right_coeffs.get(interaction, 0.0)
                combined_coeffs[interaction] = (left_val + right_val) / 2

                new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
                combined_coeffs[new_interaction] = (left_val - right_val) / 2
            return combined_coeffs

        def _dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
            """Performs a depth-first traversal of the tree to compute coefficients."""
            # Base case: if the node is a leaf, its function is a constant.
            if "leaf_value" in node:
                # The only non-zero coefficient is for the empty interaction (the bias term).
                return {(): node["leaf_value"]}
            # Recursive step: if the node is a split node.
            left_coeffs = _dfs_traverse(node["left_child"])
            right_coeffs = _dfs_traverse(node["right_child"])
            feature_idx = node["split_feature"]
            return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

        return _dfs_traverse(tree_info["tree_structure"])

    def _xgboost_to_fourier(self, model_dict: list[dict[str, Any]]) -> dict[tuple[int, ...], float]:
        """Extracts the aggregated Fourier coefficients from an XGBoost model dictionary.

        This method iterates over all trees in the XGBoost ensemble, computes the
        Fourier coefficients for each individual tree using the `_xgboost_tree_to_fourier`
        helper method, and then sums these coefficients to get the final Fourier
        representation of the complete model.

        Args:
            model_dict: A list of dictionaries representing the trained XGBoost model,
                as produced by `model.get_booster().get_dump(dump_format="json")` parsed
                as JSON.

        Returns:
            A dictionary that maps interaction tuples (representing Fourier frequencies)
            to their aggregated Fourier coefficients.
        """
        aggregated_coeffs = defaultdict(float)

        for tree_info in model_dict:
            tree_coeffs = self._xgboost_tree_to_fourier(json.loads(tree_info))
            for interaction, value in tree_coeffs.items():
                aggregated_coeffs[interaction] += value

        return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

    def _xgboost_tree_to_fourier(self, tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Recursively strips the Fourier coefficients from a single XGBoost tree.

        This method traverses a tree's structure, as provided by XGBoost's `get_dump`
        method with `dump_format="json"`, and computes the Fourier representation of the
        piecewise-constant function that the tree defines.

        XGBoost's JSON dump format differs from LightGBM's:
        - Leaf nodes have a "leaf" key (vs "leaf_value" in LGBM).
        - Split nodes use "split" for the feature name (e.g. "f0", "f1") and "children"
          as a list of child nodes where the first child (index 0) is the "yes" branch
          and the second child (index 1) is the "no" branch.

        Args:
            tree_info: A dictionary representing a single decision tree from an XGBoost
                model.

        Returns:
            A dictionary mapping interaction tuples to their corresponding coefficients
            for the single tree.

        References:
            Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
            "SHAP values via sparse Fourier representation"
            arXiv preprint arXiv:2410.06300 (2024).
        """

        def _combine_coeffs(
            left_coeffs: dict[tuple[int, ...], float],
            right_coeffs: dict[tuple[int, ...], float],
            feature_idx: int,
        ) -> dict[tuple[int, ...], float]:
            """Combines Fourier coefficients from the left and right children of a split node."""
            combined_coeffs = {}
            all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

            for interaction in all_interactions:
                left_val = left_coeffs.get(interaction, 0.0)
                right_val = right_coeffs.get(interaction, 0.0)
                combined_coeffs[interaction] = (left_val + right_val) / 2

                new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
                combined_coeffs[new_interaction] = (left_val - right_val) / 2
            return combined_coeffs

        def _dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
            """Performs a depth-first traversal of the tree to compute coefficients."""
            # Base case: leaf node in XGBoost has a "leaf" key.
            if "leaf" in node:
                return {(): node["leaf"]}

            # Recursive step: split node.
            # XGBoost uses "children" list: index 0 = yes/left, index 1 = no/right.
            left_coeffs = _dfs_traverse(node["children"][0])
            right_coeffs = _dfs_traverse(node["children"][1])

            # XGBoost stores feature as a string like "f0", "f1", etc.
            # Extract the integer index.
            feature_idx = int(node["split"][1:])

            return _combine_coeffs(left_coeffs, right_coeffs, feature_idx)

        return _dfs_traverse(tree_info)

    def _refine(
        self,
        four_dict: dict[tuple[int, ...], float],
        train_X: np.ndarray,
        train_y: np.ndarray,
    ) -> dict[tuple[int, ...], float]:
        """Refines the estimated Fourier coefficients using a Ridge regression model.

        This method takes an initial set of estimated Fourier coefficients and refines them to
        better fit the observed game values. It first identifies the most significant
        coefficients by keeping those that contribute to 95% of the total "energy" (sum of
        squared Fourier coefficients, excluding the baseline). Then, it constructs a new feature matrix
        based on the Fourier basis functions corresponding to these significant interactions.
        Finally, it fits a `RidgeCV` model to re-estimate the values of these coefficients,
        effectively fine-tuning them against the training data.

        Args:
            four_dict: A dictionary mapping interaction tuples to their initial estimated
                Fourier coefficient values.
            train_X: The training data matrix where rows are coalitions (binary vectors) and
                columns are players.
            train_y: The corresponding game values for each coalition in `train_X`.

        Returns:
            A dictionary containing the refined Fourier coefficients for the most significant
            interactions.
        """
        n = train_X.shape[1]
        four_items = list(four_dict.items())
        list_keys = [item[0] for item in four_items]
        four_coefs = np.array([item[1] for item in four_items])

        nfc_idx = list_keys.index(()) if () in list_keys else None

        four_coefs_for_energy = np.copy(four_coefs)
        if nfc_idx is not None:
            four_coefs_for_energy[nfc_idx] = 0
        four_coefs_sq = four_coefs_for_energy**2
        tot_energy = np.sum(four_coefs_sq)
        sorted_four_coefs_sq = np.sort(four_coefs_sq)[::-1]
        cumulative_energy_ratio = np.cumsum(sorted_four_coefs_sq / tot_energy)
        if self.decoder_type.endswith(".7trunjaction"):
            thresh_idx_95 = np.argmin(cumulative_energy_ratio < 0.7) + 1
            thresh = np.sqrt(sorted_four_coefs_sq[thresh_idx_95])
        elif self.decoder_type.endswith("notrunjaction"):
            thresh_idx_95 = len(cumulative_energy_ratio)
            thresh = -1
        else:
            thresh_idx_95 = np.argmin(cumulative_energy_ratio < self.cut_off_quantile) + 1
            thresh = np.sqrt(sorted_four_coefs_sq[thresh_idx_95])

        four_dict_trunc = {
            tuple(int(i in k) for i in range(n)): v for k, v in four_dict.items() if abs(v) > thresh
        }
        print(
            f"Refinement: Keeping {len(four_dict_trunc)} out of {len(four_dict)} Fourier coefficients "
            f"after applying the energy cutoff threshold."
        )
        if self.decoder_type.endswith("_norefinement"):
            print("No refinement applied, returning initial Fourier coefficients.")
            return {
                tuple(i for i, x in enumerate(k) if x): v
                for k, v in four_dict_trunc.items()
            }
        support = np.array(list(four_dict_trunc.keys()))

        X = np.real(np.exp(train_X @ (1j * np.pi * support.T))) # shape (num_samples, num_significant_fourier_terms)
        a = time.perf_counter()
        reg = RidgeCV(alphas=np.logspace(-6, 6, 100), fit_intercept=False).fit(X, train_y)

        regression_coefs = dict(
            zip([tuple(s.astype(int)) for s in support], reg.coef_, strict=False)
        )
        b = time.perf_counter()
        print(f"Ridge regression fitting time: {b - a:.4f} seconds")
        return {tuple(i for i, x in enumerate(k) if x): v for k, v in regression_coefs.items()}
