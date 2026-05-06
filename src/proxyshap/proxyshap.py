"""ProxySHAP — proxy-model approximator for higher-order Shapley interactions.

This module exposes:

* :class:`ProxySHAP` — fits a surrogate model (XGBoost or scikit-learn linear
  regressor) on sampled coalition values, then reads off interactions from the
  surrogate and optionally corrects the residual error with an adjustment
  estimator (``"none"``, ``"msr"``, ``"svarm"``, ``"kernel"``).
* :class:`ProxySHAPHPO` — :class:`ProxySHAP` variant that searches for the
  proxy's hyperparameters with SMAC before computing interactions.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from shapiq.approximator.base import Approximator
from shapiq.approximator.montecarlo.shapiq import SHAPIQ
from shapiq.approximator.montecarlo.svarmiq import SVARMIQ
from shapiq.approximator.regression.kernelshapiq import KernelSHAPIQ
from shapiq.game import Game
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer
from shapiq.utils.modules import safe_isinstance
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from smac import HyperparameterOptimizationFacade, Scenario
from xgboost import XGBRegressor

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues


def extract_linear_interactions(coefficients, max_order, n_players, poly):
    """Map coefficients of a linear-in-features model back to interaction tuples.

    Args:
        coefficients: Fitted model coefficients, ordered to match ``poly``'s
            column layout when ``max_order > 1``.
        max_order: Maximum interaction order materialized in the proxy.
        n_players: Number of players (features) in the coalition game.
        poly: Fitted :class:`sklearn.preprocessing.PolynomialFeatures` instance
            used to expand coalition matrices for ``max_order > 1``. Ignored
            when ``max_order == 1``.

    Returns:
        Mapping from interaction tuple (sorted feature indices) to its
        coefficient.
    """
    if max_order == 1:
        linear_interactions = {(i,): coefficients[i] for i in range(n_players)}
    else:
        # poly exists
        interaction_to_col = {}
        for col, p in enumerate(poly.powers_):
            interactions = np.flatnonzero(p)
            interactions.sort()
            interactions = interactions.tolist()
            idx = tuple(interactions)  # features used in this interaction
            interaction_to_col[idx] = col

        # Now build your coefficient dict safely
        linear_interactions = {idx: coefficients[col] for idx, col in interaction_to_col.items()}

    return linear_interactions


def get_configurations_smac(model_name: str, random_state: int = 42):
    """Build the SMAC :class:`ConfigurationSpace` for a supported proxy model.

    Args:
        model_name: ``"xgb"`` or ``"lightgbm"``.
        random_state: Unused; retained for API symmetry with callers.

    Returns:
        A :class:`ConfigSpace.ConfigurationSpace` covering the proxy's
        tunable hyperparameters.

    Raises:
        ValueError: If ``model_name`` is not a supported proxy.
    """
    cs = ConfigurationSpace(name=f"SMAC_{model_name}_ConfigSpace")
    if model_name == "xgb":
        # Integers
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter(
                    "n_estimators", lower=100, upper=2000, default_value=800
                ),
                UniformIntegerHyperparameter("max_depth", lower=2, upper=8, default_value=4),
                UniformIntegerHyperparameter(
                    "min_child_weight", lower=1, upper=20, default_value=5
                ),
            ]
        )

        # Floats
        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter("subsample", lower=0.4, upper=1.0, default_value=1),
                UniformFloatHyperparameter(
                    "colsample_bytree", lower=0.4, upper=1.0, default_value=1
                ),
                # learning_rate is almost always better searched on a log scale
                UniformFloatHyperparameter(
                    "learning_rate", lower=1e-3, upper=0.3, default_value=0.05, log=True
                ),
                # L2 regularization
                UniformFloatHyperparameter(
                    "reg_lambda", lower=1e-3, upper=50.0, default_value=1.0, log=True
                ),
                # L1 regularization
                UniformFloatHyperparameter(
                    "reg_alpha", lower=1e-3, upper=50.0, default_value=1.0, log=True
                ),
            ]
        )
    elif model_name == "lightgbm":
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter("max_depth", lower=2, upper=6, default_value=3),
                UniformIntegerHyperparameter(
                    "n_estimators", lower=100, upper=2000, default_value=800
                ),
                UniformIntegerHyperparameter(
                    "min_child_samples", lower=2, upper=20, default_value=20
                ),
            ]
        )

        cs.add_hyperparameters(
            [
                UniformFloatHyperparameter(
                    "learning_rate", lower=1e-2, upper=1e-1, default_value=0.1, log=True
                ),
            ]
        )
    else:
        raise ValueError(f"Model {model_name} not recognized for SMAC configuration.")

    return cs


class ProxySHAP(Approximator):
    """Proxy-model approximator for higher-order Shapley interactions.

    Two-stage estimator:

    1. **Proxy fit.** A surrogate (XGBoost regressor by default; any tree- or
       linear-proxy following the scikit-learn API) is fit on a sample of
       coalition values. Interactions are then read off the proxy exactly —
       analytically for linear proxies, and via
       :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
       for tree proxies (boolean-tree mode, since coalitions are binary).
    2. **Residual adjustment** *(optional)*. The proxy's prediction error on
       the sampled coalitions is treated as a smaller game and approximated
       with one of ``"msr"`` (SHAPIQ), ``"svarm"`` (SVARMIQ), or ``"kernel"``
       (KernelSHAPIQ). Set ``adjustment="none"`` to skip.

    The optional ``disjoint`` mode splits the sampled coalitions into a
    disjoint train/adjust split so the residual is evaluated on coalitions
    unseen by the proxy.
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: str = "SII",
        proxy_model: object | None = None,
        adjustment: str = "msr",
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = False,
        random_state: int | None = None,
        disjoint: bool = False,
    ) -> None:
        """Initialize the ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum order of interactions to consider.
            index: Interaction index to estimate (e.g. ``"SII"``, ``"k-SII"``,
                ``"FSII"``, ``"STII"``, ``"FBII"``).
            proxy_model: Surrogate model used to fit coalition values. Tree
                proxies (default ``XGBRegressor``) take the boolean-tree
                interventional path; an instance of
                :class:`sklearn.linear_model.LinearRegression` triggers the
                linear path. If ``None``, a default ``XGBRegressor`` is used.
            adjustment: Residual-adjustment estimator. One of:
                ``"none"`` (no adjustment),
                ``"msr"`` (unbiased SHAPIQ),
                ``"svarm"`` (SVARMIQ),
                ``"kernel"`` (KernelSHAPIQ).
            sampling_weights: Optional weights of shape ``(n + 1,)`` controlling
                the probability of sampling a coalition by size. Defaults to
                ``None``.
            pairing_trick: If ``True``, applies the pairing trick to the
                coalition sampler. Defaults to ``False``.
            random_state: Seed forwarded to the sampler and the default proxy.
                Defaults to ``None``.
            disjoint: If ``True``, holds out a fraction of sampled coalitions
                for the residual-adjustment step (proxy is fit only on the
                remainder). Tree path only. Defaults to ``False``.
        """
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
            initialize_dict=False,
        )
        self._sampling_weights = sampling_weights
        self._pairing_trick = pairing_trick
        if proxy_model is not None:
            self.proxy_model = proxy_model
        else:
            try:
                from xgboost import XGBRegressor
            except ImportError as e:
                msg = "XGBoost is required for the default proxy model. Please install it with 'pip install xgboost' or provide a custom proxy_model."
                raise ImportError(msg) from e
            self.proxy_model = XGBRegressor(
                random_state=random_state,
                tree_method="hist",
                n_jobs=-1,
                objective="reg:squarederror",
            )
        self.disjoint = disjoint
        self.set_adjustment_method(adjustment)

    def set_adjustment_method(self, adjustment: str) -> None:
        """Select the residual-adjustment estimator.

        Args:
            adjustment: One of ``"none"``, ``"msr"``, ``"svarm"``, ``"kernel"``.
                See :class:`ProxySHAP` for what each does.

        Raises:
            ValueError: If ``adjustment`` is not one of the accepted values.
        """
        if adjustment not in {"none", "msr", "svarm", "kernel"}:
            msg = f"Invalid adjustment method: {adjustment}"
            raise ValueError(msg)
        self.adjustment = adjustment
        match adjustment:
            case "msr":
                self.adjustment_method = SHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
                
            case "svarm":
                self.adjustment_method = SVARMIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )
            case "kernel":
                self.adjustment_method = KernelSHAPIQ(
                    n=self.n,
                    max_order=self.max_order,
                    index=self.index,
                    sampling_weights=self._sampling_weights,
                    pairing_trick=self._pairing_trick,
                    random_state=self._random_state,
                )

    def approximate(self, budget, game, **kwargs):
        """Approximate interaction values, dispatching by proxy type.

        Routes to :meth:`approximate_linear` for an
        :class:`sklearn.linear_model.LinearRegression` proxy and to
        :meth:`approximate_tree` for everything else.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game (a :class:`shapiq.game.Game` or any callable
                accepting a binary coalition matrix and returning game values).
            **kwargs: Forwarded to the dispatched method.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for orders 0
            through ``self.max_order``.
        """
        if safe_isinstance(self.proxy_model, "sklearn.linear_model.LinearRegression"):
            return self.approximate_linear(budget, game, **kwargs)
        else:
            return self.approximate_tree(budget, game, **kwargs)

    def approximate_linear(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **_: dict
    ) -> InteractionValues:
        """Approximate interactions with a linear-in-features proxy.

        For ``max_order > 1`` the coalition matrix is expanded with
        :class:`sklearn.preprocessing.PolynomialFeatures` (interaction-only) so
        that fitted coefficients map directly to Möbius interactions; the
        result is then converted to ``self.approximation_index`` via
        :class:`~shapiq.game_theory.moebius_converter.MoebiusConverter`.
        Optional residual adjustment is applied to the proxy's residuals on
        the same coalitions. Per-stage timings populate
        ``self.runtime_last_approximate_run``.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        t_start = time.time()
        # sample with current budget
        self._sampler.sample(int(budget))
        coalitions_matrix = self._sampler.coalitions_matrix
        coalitions_matrix_binary = coalitions_matrix.copy()
        timed_game, get_eval_time = self._make_timed_game(game)
        coalition_values = timed_game(coalitions_matrix)
        baseline_value = coalition_values[0]
        coalition_values -= baseline_value

        first_stage_start = time.perf_counter()
        if self.max_order == 1:
            poly = None
            pass  # no expansion needed for linear model
        else:
            from sklearn.preprocessing import PolynomialFeatures

            poly = PolynomialFeatures(
                degree=self.max_order, interaction_only=True, include_bias=False
            )
            coalitions_matrix = poly.fit_transform(coalitions_matrix)
        self.proxy_model.fit(coalitions_matrix, coalition_values)

        # Extract the coeffiecients as interaction values
        linear_interactions = extract_linear_interactions(
            self.proxy_model.coef_, self.max_order, self.n, poly
        )

        proxy_interactions = InteractionValues(
            linear_interactions,
            index=self.approximation_index,
            n_players=self.n,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=float(baseline_value),
            estimated=not budget >= 2**self.n,
            estimation_budget=int(budget),
        )
        proxy_interactions = MoebiusConverter(moebius_coefficients=proxy_interactions).compute(
            index=self.approximation_index, order=self.max_order
        )
        first_stage_end = time.perf_counter()

        if self.adjustment != "none":
            residual_values = coalition_values - self.proxy_model.predict(  # ty: ignore[unresolved-attribute]
                coalitions_matrix
            )
            residual_values -= residual_values[0]  # Normalize residuals
            proxy_interactions += self.adjustment_method.approximate_given(
                coalitions_matrix=coalitions_matrix_binary,
                game_values=residual_values,
                sampling_adjustment_weights=self._sampler.sampling_adjustment_weights,
            )
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct
        adjustment_end_time = time.perf_counter()
        self.runtime_last_approximate_run = {
            "evaluations": get_eval_time(),
            "proxy_time": first_stage_end - first_stage_start,
            "adjustment_time": adjustment_end_time - first_stage_end,
            "total": time.perf_counter() - t_start,
        }
        return proxy_interactions

    def approximate_tree(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray], **_: dict
    ) -> InteractionValues:
        """Approximate interactions with a tree proxy and exact tree readout.

        Samples ``budget`` coalitions, evaluates the game, fits the tree proxy,
        then reads off interactions exactly via
        :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
        in boolean-tree mode. If ``self.disjoint`` is set, sampled coalitions
        are split (80/20) so the proxy is fit on one part and the residual
        adjustment is computed on the other. Per-stage timings populate
        ``self.runtime_last_approximate_run``.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        t_start = time.perf_counter()
        # 1. Sample coalitions and fit proxy tree
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        
        if self.disjoint:
            timed_game, get_eval_time = self._make_timed_game(game)
            game_values = timed_game(coalitions_matrix)
            baseline_value = game_values[0]  # Value of the empty coalition
            game_values -= baseline_value  # Normalize values
            coal_matrix_train, coal_matrix_adjust, coalition_values_train, coalition_values_adjust, sampling_adjustment_weights_train, sampling_adjustment_weights_adjust = train_test_split(
                coalitions_matrix, game_values, self._sampler.sampling_adjustment_weights, test_size=0.2, random_state=self._random_state
            )

            self.proxy_model.fit(coal_matrix_train, coalition_values_train)  # ty: ignore[unresolved-attribute]
            explainer = InterventionalTreeExplainer(
                self.proxy_model,
                data=np.zeros((1, self.n)),  # reference data for boolean tree
                class_index=None,
                index=self.index,
                max_order=self.max_order,
                bool_tree=True,
            )
            proxy_values = explainer.explain_function(np.ones((1, self.n)))
            proxy_interactions = InteractionValues(
                values=proxy_values.interactions,
                index=self.index,
                max_order=self.max_order,
                n_players=self.n,
                min_order=0,
                estimated=budget >= 2**self.n,
                estimation_budget=budget,
                baseline_value=float(baseline_value),
            )
            if self.adjustment != "none":
                prediction_tree = self.proxy_model.predict(coal_matrix_adjust)  # ty: ignore[unresolved-attribute]
                residuals = coalition_values_adjust - prediction_tree
                adjustment_values = self.adjustment_method.approximate_given(
                    coalitions_matrix=coal_matrix_adjust,
                    game_values=residuals,
                    sampling_adjustment_weights=sampling_adjustment_weights_adjust,
                )
                proxy_interactions += adjustment_values
            proxy_interactions.baseline_value = baseline_value
            proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct
            adjustment_end_time = time.perf_counter()
            self.runtime_last_approximate_run = {
                "evaluations": get_eval_time(),
                "proxy_time": adjustment_end_time - t_start,
                "adjustment_time": 0,  # Adjustment time is included in proxy_time since we fit the proxy model on the training set and compute adjustment on the adjustment set without refitting              
                "total": time.perf_counter() - t_start, 
            }
            return proxy_interactions
        else:
            timed_game, get_eval_time = self._make_timed_game(game)
            coalition_values = timed_game(coalitions_matrix)
            baseline_value = coalition_values[0]  # Value of the empty coalition
            coalition_values -= baseline_value  # Normalize values
            first_stage_start = time.perf_counter()
            self.proxy_model.fit(  # ty: ignore[unresolved-attribute]
                coalitions_matrix, coalition_values
            )  
            if safe_isinstance(self.proxy_model, "sklearn.model_selection._search.GridSearchCV"):
                self.proxy_model = self.proxy_model.best_estimator_
                
                
            # 2. Compute exact index&max_order for the proxy model
            explainer = InterventionalTreeExplainer(
                self.proxy_model,
                data=np.zeros((1, self.n)),  # reference data for boolean tree
                class_index=None,
                index=self.index,
                max_order=self.max_order,
                bool_tree=True,
            )
            proxy_values = explainer.explain_function(np.ones((1, self.n)))
            proxy_interactions = InteractionValues(
                values=proxy_values.interactions,
                index=self.index,
                max_order=self.max_order,
                n_players=self.n,
                min_order=0,
                estimated=budget >= 2**self.n,
                estimation_budget=budget,
                baseline_value=float(baseline_value),
            )
            first_stage_end = time.perf_counter()
            if self.adjustment != "none":
                residual_values = coalition_values - self.proxy_model.predict(  # ty: ignore[unresolved-attribute]
                    coalitions_matrix
                )
                residual_values -= residual_values[0]  # Normalize residuals
                proxy_interactions += self.adjustment_method.approximate_given(
                    coalitions_matrix=coalitions_matrix,
                    game_values=residual_values,
                    sampling_adjustment_weights=self._sampler.sampling_adjustment_weights,
                )
            proxy_interactions.baseline_value = baseline_value
            proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct
            adjustment_end_time = time.perf_counter()
            self.runtime_last_approximate_run = {
                "evaluations": get_eval_time(),
                "proxy_time": first_stage_end - first_stage_start,
                "adjustment_time": adjustment_end_time - first_stage_end,
                "total": time.perf_counter() - t_start,
            }
            return proxy_interactions

    def approximate_given(self, coalitions_matrix: CoalitionMatrix, game_values: GameValues, baseline_value: float, sampling_adjustment_weights: FloatVector | None = None) -> InteractionValues:
        """Run the tree proxy + adjustment pipeline on caller-supplied samples.

        Skips coalition sampling and game evaluation: callers pass the
        coalitions, their game values, and the empty-coalition baseline
        directly. Useful when ProxySHAP is composed inside a larger pipeline
        that has already drawn its own coalitions.

        Args:
            coalitions_matrix: Binary coalition matrix of shape
                ``(n_samples, n_players)``.
            game_values: Game values for each coalition, shape ``(n_samples,)``.
            baseline_value: Game value of the empty coalition; written into the
                returned :class:`~shapiq.interaction_values.InteractionValues`
                and used as the order-0 entry.
            sampling_adjustment_weights: Optional per-coalition weights forwarded
                to the residual-adjustment estimator. Defaults to ``None``.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        self.proxy_model.fit(coalitions_matrix, game_values)  # ty: ignore[unresolved-attribute]
        explainer = InterventionalTreeExplainer(
            self.proxy_model,
            data=np.zeros((1, self.n)),  # reference data for boolean tree
            class_index=None,
            index=self.index,
            max_order=self.max_order,
            bool_tree=True,
        )
        proxy_values = explainer.explain_function(np.ones((1, self.n)))
        proxy_interactions = InteractionValues(
            values=proxy_values.interactions,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n,
            min_order=0,
            estimated=True,
            estimation_budget=coalitions_matrix.shape[0],
            baseline_value=float(baseline_value),
        )
        if self.adjustment != "none":
                residual_values = game_values - self.proxy_model.predict(  # ty: ignore[unresolved-attribute]
                    coalitions_matrix
                )
                residual_values -= residual_values[0]  # Normalize residuals
                proxy_interactions += self.adjustment_method.approximate_given(
                    coalitions_matrix=coalitions_matrix,
                    game_values=residual_values,
                    sampling_adjustment_weights=sampling_adjustment_weights,
                )
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions.interactions[()] = baseline_value  # Ensure empty coalition value is correct
        return proxy_interactions
class ProxySHAPHPO(ProxySHAP):
    """:class:`ProxySHAP` variant with SMAC hyperparameter search on the proxy.

    Each call to :meth:`approximate` runs a SMAC
    :class:`~smac.HyperparameterOptimizationFacade` (200 trials, 5-fold CV with
    negative MSE) over the proxy's configuration space, refits the best
    configuration on all sampled coalitions, then proceeds with the same exact
    tree readout + optional residual adjustment as the base
    :class:`ProxySHAP`. SMAC artefacts are written under
    ``smac_config_save_path / "smac_output" / ...``.
    """

    def __init__(
        self,
        n: int,
        *,
        max_order: int = 2,
        index: str = "SII",
        proxy_model: str,
        adjustment: str = "msr",
        sampling_weights: FloatVector | None = None,
        pairing_trick: bool = False,
        random_state: int | None = None,
        smac_config_save_path: str | None = None,
    ) -> None:
        """Initialize the SMAC-tuned ProxySHAP approximator.

        Args:
            n: Number of features (players).
            max_order: Maximum order of interactions to consider.
            index: Interaction index to estimate.
            proxy_model: Name of the proxy family to tune; ``"xgb"`` or
                ``"lightgbm"``. The corresponding configuration space is
                returned by :func:`get_configurations_smac`.
            adjustment: Residual-adjustment estimator. See :class:`ProxySHAP`
                for accepted values.
            sampling_weights: Optional weights of shape ``(n + 1,)`` for the
                coalition sampler. Defaults to ``None``.
            pairing_trick: If ``True``, applies the pairing trick to the
                sampler. Defaults to ``False``.
            random_state: Seed forwarded to the sampler, the SMAC scenario,
                and the refit proxy. Defaults to ``None``.
            smac_config_save_path: Directory under which per-game SMAC outputs
                are written. Defaults to ``"smac_configs"``.
        """
        super().__init__(
            n=n,
            max_order=max_order,
            index=index,
            proxy_model=None,  # We will set the proxy model after HPO
            adjustment=adjustment,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            random_state=random_state,
        )
        self.conf_space = get_configurations_smac(proxy_model, random_state=random_state)
        self.proxy_model_name = proxy_model
        self.smac_config_save_path = smac_config_save_path

    def approximate(self, budget, game, game_id, id_explain, **_):
        """Tune the proxy with SMAC, then approximate interactions.

        Samples ``budget`` coalitions, evaluates the game, runs a SMAC
        hyperparameter search (200 trials, 5-fold CV with negative MSE) on the
        proxy's configuration space, refits the incumbent on all samples, and
        finally reads off interactions via
        :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
        with optional residual adjustment. Per-stage timings (including ``hpo``)
        populate ``self.runtime_last_approximate_run``.

        Args:
            budget: Number of coalition evaluations to draw.
            game: Coalition game.
            game_id: Identifier used to name the per-call SMAC output directory.
            id_explain: Identifier used to name the per-call SMAC output
                directory.

        Returns:
            :class:`~shapiq.interaction_values.InteractionValues` for the
            requested index and order.
        """
        t_start = time.perf_counter()

        # 1. Sample coalitions and fit proxy tree
        self._sampler.sample(budget)
        coalitions_matrix = self._sampler.coalitions_matrix
        timed_game, get_eval_time = self._make_timed_game(game)
        coalition_values = timed_game(coalitions_matrix)
        baseline_value = coalition_values[0]  # Value of the empty coalition
        coalition_values -= baseline_value  # Normalize values
        first_stage_start = time.perf_counter()

        base_path = self.smac_config_save_path or Path("smac_configs")
        smac_save_path = (
            base_path / f"smac_output/{game.__class__.__name__}-{game_id}-{id_explain}/{budget}"
        )
        smac_save_path.mkdir(parents=True, exist_ok=True)
        scenario = Scenario(
            self.conf_space,
            deterministic=True,
            n_trials=200,
            output_directory=smac_save_path,
            seed=self._random_state,
        )
        if self.proxy_model_name == "xgb":
            # define training function for smac
            def train(config: Configuration, seed: int = 0) -> float:
                params = dict(config)

                model = XGBRegressor(
                    objective="reg:squarederror",
                    random_state=seed,
                    tree_method="hist",
                    n_jobs=int(
                        os.environ.get("OMP_NUM_THREADS", -1)
                    ),  # avoid nested parallelism with CV/SMAC
                    **params,
                )

                cv = KFold(n_splits=5, shuffle=True, random_state=seed)
                scores = cross_val_score(
                    model,
                    coalitions_matrix,
                    coalition_values,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,  # avoid nested parallelism with CV/SMAC
                )
                return -float(np.mean(scores))  # MSE loss

        elif self.proxy_model_name == "lightgbm":
            # define training function for smac
            def train(config: Configuration, seed: int = 0) -> float:
                import lightgbm as lgb

                params = dict(config)

                model = lgb.LGBMRegressor(
                    random_state=seed,
                    n_jobs=int(
                        os.environ.get("OMP_NUM_THREADS", -1)
                    ),  # avoid nested parallelism with CV/SMAC
                    **params,
                )

                cv = KFold(n_splits=5, shuffle=True, random_state=seed)
                scores = cross_val_score(
                    model,
                    coalitions_matrix,
                    coalition_values,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,  # avoid nested parallelism with CV/SMAC
                )
                return -float(np.mean(scores))  # MSE loss

        # perform hyperparameter optimization using smac
        smac = HyperparameterOptimizationFacade(scenario, train, overwrite=False)
        incumbent = smac.optimize()
        hpo_end_time = time.time()
        # get the best model parameters
        best_params = dict(incumbent)
        print("Best hyperparameters found by SMAC: ", best_params)
        if self.proxy_model_name == "xgb":
            self.value_model = XGBRegressor(
                objective="reg:squarederror",
                random_state=self._random_state,
                tree_method="hist",
                n_jobs=int(
                    os.environ.get("OMP_NUM_THREADS", -1)
                ),  # avoid nested parallelism with CV/SMAC
                **best_params,
            )
        elif self.proxy_model_name == "lightgbm":
            import lightgbm as lgb

            self.value_model = lgb.LGBMRegressor(
                random_state=self._random_state,
                n_jobs=int(
                    os.environ.get("OMP_NUM_THREADS", -1)
                ),  # avoid nested parallelism with CV/SMAC
                **best_params,
            )
        else:
            raise ValueError("Unknown value_model for SMAC optimization.")
        self.proxy_model = self.value_model  # Set the proxy model to the best model found by HPO
        # Fit the proxy model on the entire dataset
        self.proxy_model.fit(coalitions_matrix, coalition_values)  # ty: ignore[unresolved-attribute]
        first_stage_end = time.perf_counter()
        # 2. Compute exact index&max_order for the proxy model
        explainer = InterventionalTreeExplainer(
            self.proxy_model,
            data=np.zeros((1, self.n)),  # reference data for boolean tree
            class_index=None,
            index=self.index,
            max_order=self.max_order,
            bool_tree=True,
        )
        proxy_values = explainer.explain_function(np.ones((1, self.n)))
        proxy_interactions = InteractionValues(
            values=proxy_values.interactions,
            index=self.index,
            max_order=self.max_order,
            n_players=self.n,
            min_order=0,
            estimated=budget >= 2**self.n,
            estimation_budget=budget,
            baseline_value=float(baseline_value),
        )
        first_stage_end = time.perf_counter()
        if self.adjustment != "none":
            residual_values = coalition_values - self.proxy_model.predict(  # ty: ignore[unresolved-attribute]
                coalitions_matrix
            )
            residual_values -= residual_values[0]  # Normalize residuals
            proxy_interactions += self.adjustment_method.approximate_given(
                coalitions_matrix=coalitions_matrix,
                game_values=residual_values,
                sampling_adjustment_weights=self._sampler.sampling_adjustment_weights,
            )
        proxy_interactions.baseline_value = baseline_value
        proxy_interactions[()] = baseline_value  # Ensure empty coalition value is correct
        adjustment_end_time = time.perf_counter()
        self.runtime_last_approximate_run = {
            "evaluations": get_eval_time(),
            "proxy_time": first_stage_end - first_stage_start,
            "hpo": hpo_end_time - first_stage_start,
            "adjustment_time": adjustment_end_time - first_stage_end,
            "total": time.perf_counter() - t_start,
        }
        return proxy_interactions
