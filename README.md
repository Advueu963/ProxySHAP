# Proxy-Based Approximation of Shapley and Banzhaf Interactions

Code accompanying the NeurIPS submission. This repository contributes two methods for
computing higher-order Shapley interactions:

- **ProxySHAP** — a two-stage proxy-model approximator. A surrogate (XGBoost by default,
  optionally a linear regressor) is fit on a sample of coalition values; interactions are
  read off the surrogate exactly, and the residual is optionally corrected with a smaller
  Shapley-interaction estimator (`msr`, `svarm`, or `kernel`). Implemented in
  [src/proxyshap/proxyshap.py](src/proxyshap/proxyshap.py) (`ProxySHAP`, `ProxySHAPHPO`).
- **InterventionalTreeExplainer** — an exact, any-order interventional Shapley-interaction
  explainer for tree models, with C++ kernels for both the dense low-order regime and the
  sparse high-order regime. Implemented in
  [shapiq_local/src/shapiq/tree/interventional/explainer.py](shapiq_local/src/shapiq/tree/interventional/explainer.py).

## Repository layout

| Path | Contents |
| --- | --- |
| [src/proxyshap/](src/proxyshap/) | The `proxyshap` package (ProxySHAP approximator and helpers). |
| [shapiq_local/](shapiq_local/) | Local fork of [shapiq](https://github.com/mmschlk/shapiq) v1.4.2; adds `InterventionalTreeExplainer` under `src/shapiq/tree/interventional/`. |
| [shapiq-benchmark/](shapiq-benchmark/) | Benchmarking framework. JSON experiment configs in `benchmarks/`; approximator, game, and metric utilities in `src/shapiq_benchmark/`. |
| [experiments/](experiments/) | Runnable scripts: precomputation, sequential / SLURM benchmarks, hyperparameter search. |
| [ground_truth/](ground_truth/) | Cached exact Shapley values per game type (`exhaustive`, `interventional`, `pathdependent`). |
| [approximations/](approximations/) | Output directory written by the benchmark scripts. |

## Installation

This is a [`uv`](https://docs.astral.sh/uv/) workspace (Python ≥ 3.12); members are
`shapiq-benchmark` and `shapiq_local`. A single command installs everything:

```bash
uv sync
```


## Reproducing the results

The pipeline has five stages. Replace `<config.json>` with one of the JSON files under
[shapiq-benchmark/benchmarks/](shapiq-benchmark/benchmarks/) (named
`configuration_{game_type}_{index}_order{n}.json`); `<game_type>` is one of
`exhaustive`, `faithful`, `interventional`, `pathdependent`.

**1. Pre-compute coalition games.** Caches model predictions per coalition so they are
not recomputed across approximators:

```bash
uv run python experiments/precompute_games.py
```

**2. Compute ground truth.**

```bash
uv run python experiments/benchmark_local.py \
    --config <config.json> --game_type <game_type> --mode true
```

**3. Run approximators.** This is the main experiment loop and produces the curves
reported in the paper:

```bash
uv run python experiments/benchmark_local.py \
    --config <config.json> --game_type <game_type> --mode approx \
    --config_approximators 37
```

`--config_approximators` selects an approximator bundle: `37` is PAIRING + no-replacement
(the default), `38` is PAIRING + replacement, and `39`/`40` are the no-PAIRING variants.
A SLURM-array equivalent is provided as
[experiments/benchmark_slurm.py](experiments/benchmark_slurm.py).

**4. Aggregate metrics.**

```bash
python computation_of_approximation_metrics_local.py \
    --config_approximators 37 --config shapiq-benchmark/benchmarks/tabarena/configuration_interventional_tabarena_shapley_order2_medium.json --index SII --order 2 --game_type interventional
```

**5. Plot.**

```bash
python plot_approximation.py --game_type interventional
```
For the paper specific plots we refer to `plotting_scripts`.
Due to the .csvs exceeding more than 100MB we provide for each dataset the plots in `plots/`.
There you can also find those used in the main paper.

## Using the contributions directly

Both contributions can be used outside the benchmarking harness.

**InterventionalTreeExplainer** — exact any-order interventional interactions for any
scikit-learn / XGBoost / LightGBM tree (or ensemble):

```python
from shapiq.tree import InterventionalTreeExplainer

# `model` is a fitted tree or tree ensemble; `X_background` is the reference dataset.
explainer = InterventionalTreeExplainer(
    model=model,
    data=X_background,
    max_order=2,
    index="SII",
)
interactions = explainer.explain_function(x_explain)  # x_explain shape: (n_features,)
```

**ProxySHAP** — proxy-model approximator for arbitrary value functions:

```python
from proxyshap.proxyshap import ProxySHAP

approximator = ProxySHAP(
    n=n_features,
    max_order=2,
    index="SII",
    adjustment="msr",   # "none" | "msr" | "svarm" | "kernel"
    pairing_trick=True,
    random_state=0,
)
interactions = approximator.approximate(budget=2048, game=game)
```

`game` is any callable mapping a binary coalition matrix of shape
`(batch, n_features)` to a vector of game values, or a `shapiq.game.Game` instance.


## Acknowledgements

[shapiq_local/](shapiq_local/) is a fork of [shapiq](https://github.com/mmschlk/shapiq)
v1.4.2; refer to the upstream license bundled in that subdirectory.
