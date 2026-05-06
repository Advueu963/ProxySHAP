"""Local graph benchmark.

Runs Shapley-interaction approximators on `GraphGame` instances built from a pre-trained
GNN and TU graph dataset (defaults: GIN / Mutagenicity / n_layers=2). Ground truth is
computed via the GraphSHAP-IQ sparsity routine — no 2**n enumeration — so n_players
above 14 is fine.

Outputs follow `experiments/benchmark_local.py`'s layout so existing plotting can pick
them up with `--game_type graph`:
    ground_truth/graph/{game_id}_{rs}_{id_explain}_{index}_{order}_exact_values.json
    approximations/graph/{game_id}_{cfg}_{id_explain}_{approx}_{budget}_{index}_{order}.json

Usage:
    uv run python experiments/benchmark_graph_local.py --mode true
    uv run python experiments/benchmark_graph_local.py --mode approx --max_budget 35000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset

from shapiq import Game, InteractionValues
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.utils import powerset
from shapiq_benchmark.approximators import get_approximators

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPHSHAPIQ_ROOT = REPO_ROOT / "graphshapiq"
GNN_PRED_FILE = (
    GRAPHSHAPIQ_ROOT
    / "shapiq"
    / "explainer"
    / "graph"
    / "graph_models"
    / "graph_prediction.py"
)
TU_DATA_ROOT = GRAPHSHAPIQ_ROOT / "shapiq" / "explainer" / "graph" / "graph_datasets"
CKPT_ROOT = (
    GRAPHSHAPIQ_ROOT / "shapiq" / "explainer" / "graph" / "ckpt" / "graph_prediction"
)


# ─────────────────────────────────────────────────────────────────────────────
# GNN loading (imports GraphSHAP-IQ's `graph_prediction.py` as a private module
# to avoid clashing with the workspace `shapiq` package)
# ─────────────────────────────────────────────────────────────────────────────


def _load_gnn_module():
    spec = importlib.util.spec_from_file_location("_graphsiq_gnn", GNN_PRED_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_graphsiq_gnn"] = mod
    spec.loader.exec_module(mod)
    return mod


_HYPERPARAMS: dict[tuple[str, str, int], dict] = {
    ("GIN", "Mutagenicity", 1): {"hidden": 128},
    ("GIN", "Mutagenicity", 2): {"hidden": 32},
    ("GIN", "Mutagenicity", 3): {"hidden": 32},
    ("GIN", "Benzene", 2): {"hidden": 128},
}
_DEFAULT_FLAGS = {
    "node_bias": True,
    "graph_bias": True,
    "dropout": True,
    "batch_norm": True,
    "jumping_knowledge": True,
    "deep_readout": False,
}


def load_pretrained_gnn(
    model_type: str,
    dataset_name: str,
    n_layers: int,
    num_node_features: int,
    num_classes: int,
    device: torch.device,
):
    gnn_module = _load_gnn_module()
    GNN = gnn_module.GNN
    key = (model_type, dataset_name, n_layers)
    if key not in _HYPERPARAMS:
        msg = f"No hyperparameter entry for {key}. Add it to `_HYPERPARAMS`."
        raise KeyError(msg)
    params = {**_DEFAULT_FLAGS, **_HYPERPARAMS[key]}
    model = GNN(
        model_type=model_type,
        in_channels=num_node_features,
        hidden_channels=params["hidden"],
        out_channels=num_classes,
        n_layers=n_layers,
        node_bias=params["node_bias"],
        graph_bias=params["graph_bias"],
        dropout=params["dropout"],
        batch_norm=params["batch_norm"],
        jumping_knowledge=params["jumping_knowledge"],
        deep_readout=params["deep_readout"],
    ).to(device)
    model.node_model.to(device)
    model_id = "_".join(
        [
            model_type,
            dataset_name,
            str(n_layers),
            str(params["node_bias"]),
            str(params["graph_bias"]),
            str(params["hidden"]),
            str(params["dropout"]),
            str(params["batch_norm"]),
            str(params["jumping_knowledge"]),
        ]
    )
    if params["deep_readout"]:
        model_id += "_DR"
    ckpt_path = CKPT_ROOT / model_type / dataset_name / f"{model_id}.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, model_id


# ─────────────────────────────────────────────────────────────────────────────
# GraphGame
# ─────────────────────────────────────────────────────────────────────────────


class GraphGame(Game):
    """GNN local-explanation game with feature-removal node masking."""

    def __init__(
        self,
        model: torch.nn.Module,
        x_graph: Data,
        *,
        class_id: int,
        max_neighborhood_size: int,
        baseline: torch.Tensor | None = None,
        instance_id: int,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.max_neighborhood_size = max_neighborhood_size
        self.x_graph = x_graph.clone()
        self.baseline = baseline
        self.edge_index_np = self.x_graph.edge_index.detach().cpu().numpy()
        self.y_index = int(class_id)
        n_players = int(x_graph.num_nodes)

        normalization_value = float(
            self._raw_value_function(np.zeros(n_players, dtype=np.int8))[0]
        )
        super().__init__(
            n_players=n_players,
            normalize=True,
            normalization_value=normalization_value,
        )
        self.game_id = f"{type(self).__name__}_{instance_id}"

    def _mask_input(self, coalition: np.ndarray) -> Data:
        x_masked = self.x_graph.clone()
        c = torch.tensor(
            coalition.reshape(-1, 1), dtype=torch.float32, device=x_masked.x.device
        )
        if self.baseline is None:
            x_masked.x = x_masked.x * c
        else:
            base = self.baseline.to(x_masked.x.device)
            x_masked.x = x_masked.x * c + base * (1.0 - c)
        return x_masked

    def _raw_value_function(self, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        graphs = [self._mask_input(c.astype(np.int8)) for c in coalitions]
        batch = Batch.from_data_list(graphs).to(self.device)
        with torch.no_grad():
            preds = self.model(batch.x, batch.edge_index, batch.batch)
        return preds.detach().cpu().numpy()[:, self.y_index]

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self._raw_value_function(coalitions)


def compute_baseline_value(x_graph: Data) -> torch.Tensor:
    return x_graph.x.mean(0)


# ─────────────────────────────────────────────────────────────────────────────
# Exact ground truth via GraphSHAP-IQ sparsity (k-hop neighborhood enumeration)
# ─────────────────────────────────────────────────────────────────────────────


def _k_hop_neighborhoods(
    edge_index: np.ndarray, n_nodes: int, k: int
) -> dict[int, tuple[int, ...]]:
    """Return a dict mapping node → tuple of nodes within `k` hops (inclusive)."""
    # build adjacency once
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj[u].add(v)

    neighbors: dict[int, tuple[int, ...]] = {}
    for node in range(n_nodes):
        seen = {node}
        frontier = {node}
        for _ in range(k):
            nxt: set[int] = set()
            for u in frontier:
                nxt |= adj[u] - seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        neighbors[node] = tuple(sorted(seen))
    return neighbors


def graph_shapiq_exact(
    game: GraphGame,
    *,
    index: str = "k-SII",
    order: int = 2,
    max_total_budget: int = 2**20,
) -> tuple[InteractionValues, int]:
    """Compute exact k-SII (or other index) by exploiting k-hop sparsity.

    Sets `max_interaction_size = max_size_neighbors` so the efficiency routine is a no-op
    (all neighborhood subsets are enumerated). Returns (interaction_values, n_model_calls).
    """
    n = game.n_players
    neighbors = _k_hop_neighborhoods(game.edge_index_np, n, game.max_neighborhood_size)
    max_size_neighbors = max(len(s) for s in neighbors.values())

    # Collect Möbius coalitions: ⋃_node powerset(neighbors[node]).
    moebius_set: set[tuple[int, ...]] = set()
    for node_neighbors in neighbors.values():
        for s in powerset(node_neighbors, max_size=max_size_neighbors):
            moebius_set.add(tuple(s))

    if len(moebius_set) > max_total_budget:
        msg = (
            f"GraphSHAP-IQ exact budget = {len(moebius_set)} exceeds limit "
            f"{max_total_budget} (max_size_neighbors={max_size_neighbors})."
        )
        raise RuntimeError(msg)

    moebius_list = sorted(moebius_set, key=lambda t: (len(t), t))
    moebius_lookup = {s: i for i, s in enumerate(moebius_list)}

    coal_matrix = np.zeros((len(moebius_list), n), dtype=np.int8)
    for i, s in enumerate(moebius_list):
        if s:
            coal_matrix[i, list(s)] = 1

    # Single batched call; Game.__call__ already subtracts the normalization value.
    masked_predictions = game(coal_matrix)
    n_model_calls = coal_matrix.shape[0]

    # Möbius transform via inclusion-exclusion.
    moebius_values = np.zeros(len(moebius_list), dtype=float)
    for i, S in enumerate(moebius_list):
        v = 0.0
        for L in powerset(S):
            j = moebius_lookup[tuple(L)]
            v += ((-1) ** (len(S) - len(L))) * masked_predictions[j]
        moebius_values[i] = v

    sparsify_threshold = 1e-10
    keep = np.abs(moebius_values) > sparsify_threshold
    if not keep.any():
        keep[0] = True
    kept_coals = [s for s, k in zip(moebius_list, keep, strict=False) if k]
    kept_lookup = {s: i for i, s in enumerate(kept_coals)}
    kept_values = moebius_values[keep]
    baseline = float(kept_values[kept_lookup[()]]) if () in kept_lookup else 0.0

    moebius = InteractionValues(
        values=kept_values,
        interaction_lookup=kept_lookup,
        min_order=0,
        max_order=n,
        n_players=n,
        index="Moebius",
        baseline_value=baseline,
        estimated=False,
    )
    converter = MoebiusConverter(moebius_coefficients=moebius)
    interactions = converter.compute(index=index, order=order)
    interactions.estimated = False
    interactions.estimation_budget = n_model_calls
    return interactions, n_model_calls


# ─────────────────────────────────────────────────────────────────────────────
# Game selection
# ─────────────────────────────────────────────────────────────────────────────


def select_graph_indices(
    dataset: TUDataset, *, min_players: int, max_players: int, n_graphs: int, seed: int
) -> list[int]:
    rng = np.random.RandomState(seed)
    candidates = [
        i
        for i in range(len(dataset))
        if min_players <= int(_unwrap_graph(dataset[i]).num_nodes) <= max_players
    ]
    if len(candidates) < n_graphs:
        msg = (
            f"Only {len(candidates)} graphs in [{min_players},{max_players}], "
            f"requested {n_graphs}."
        )
        raise RuntimeError(msg)
    return list(rng.choice(candidates, size=n_graphs, replace=False).astype(int))


def select_graph_indices_exact(
    dataset: TUDataset, *, n_players: int, n_runs: int, seed: int
) -> list[int]:
    rng = np.random.RandomState(seed)
    candidates = [
        i for i in range(len(dataset)) if int(_unwrap_graph(dataset[i]).num_nodes) == n_players
    ]
    if len(candidates) < n_runs:
        msg = (
            f"Only {len(candidates)} graphs with n_players={n_players}, requested {n_runs}."
        )
        raise RuntimeError(msg)
    return list(rng.choice(candidates, size=n_runs, replace=False).astype(int))


def build_game(
    model: torch.nn.Module,
    dataset: TUDataset,
    data_id: int,
    n_layers: int,
    device: torch.device,
) -> GraphGame:
    x_graph = dataset[data_id]
    x_graph = _unwrap_graph(x_graph)
    return GraphGame(
        model=model,
        x_graph=x_graph,
        class_id=int(x_graph.y.item()),
        max_neighborhood_size=n_layers,
        baseline=compute_baseline_value(x_graph),
        instance_id=data_id,
        device=device,
    )


def _unwrap_graph(x):
    """If dataset returns (graph, explanation) tuple, return the graph part; else return x."""
    try:
        # tuple-like from GraphXAI: (graph, explanation)
        if isinstance(x, tuple) or isinstance(x, list):
            return x[0]
    except Exception:
        pass
    return x


# ─────────────────────────────────────────────────────────────────────────────
# CLI / main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local graph benchmark with GraphSHAP-IQ exact GT."
    )
    parser.add_argument("--mode", choices=["approx", "true"], default="approx")
    parser.add_argument("--dataset", default="Mutagenicity")
    parser.add_argument("--model_type", default="GIN")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_graphs", type=int, default=30)
    parser.add_argument(
        "--min_players", type=int, default=30, help="min players (nodes) in graph"
    )
    parser.add_argument(
        "--max_players", type=int, default=40, help="max players (nodes) in graph"
    )
    parser.add_argument(
        "--n_players",
        type=int,
        default=None,
        help="If set, select graphs with exactly this number of players (nodes)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Number of runs / molecules to sample when --n_players is set",
    )
    parser.add_argument("--index", default="SII")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--max_budget", type=int, default=35000)
    parser.add_argument("--n_budget_steps", type=int, default=20)
    parser.add_argument("--config_approximators", type=int, default=37)
    parser.add_argument("--random_state", type=int, default=40)
    parser.add_argument("--graph_seed", type=int, default=1234)
    parser.add_argument("--exact_max_budget", type=int, default=2**20)
    parser.add_argument(
        "--approximators",
        nargs="+",
        default=[
            # "KernelSHAPIQ",
            # "PermutationSamplingSII",
            # "SHAPIQ",
            # "SVARMIQ",
            "ProxySHAP (XGBoost)",
            "ProxySHAP (XGBoost, MSR)",
            #"ProxySPEX (XGBoost)",
            "ProxySHAP (Linear)",
            "ProxySHAP (Linear, MSR)"
        ],
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--override", action="store_true")
    return parser.parse_args()


def resolve_base_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if "SCRATCH_DSS" in os.environ:
        return Path(os.environ["SCRATCH_DSS"]) / "msr_int_iq"
    return REPO_ROOT


def resolve_pairing(config_id: int) -> bool:
    pairing_map = {37: True, 38: True, 39: False, 40: False}
    return pairing_map[config_id]


def run_true(
    args: argparse.Namespace,
    model,
    dataset,
    data_ids,
    truth_dir: Path,
    device,
    molecule_ids: list[int] | None = None,
) -> None:
    for id_explain, data_id in enumerate(data_ids):
        game = build_game(model, dataset, data_id, args.n_layers, device)
        # If molecule_ids provided, use dataset + molecule index naming as requested
        if molecule_ids is not None:
            mol_id = molecule_ids[id_explain]
            game_id = f"{args.dataset}_n{args.n_players}"
        else:
            game_id = f"{args.model_type}_{args.dataset}_{args.n_layers}_data{data_id}"
        save_path = (
            truth_dir
            / f"{game_id}_{args.random_state}_{id_explain}_{args.index}_{args.order}_exact_values.json"
        )
        if save_path.exists() and not args.override:
            print(f"[SKIP] {save_path.name}")
            continue
        print(
            f"[TRUE] {game_id} (n_players={game.n_players}) computing exact via GraphSHAP-IQ ..."
        )
        t0 = time.time()
        try:
            iv, n_calls = graph_shapiq_exact(
                game,
                index=args.index,
                order=args.order,
                max_total_budget=args.exact_max_budget,
            )
        except RuntimeError as exc:
            print(f"  [SKIP] {exc}")
            continue
        iv.save(save_path)
        print(
            f"  saved {save_path.name} | n_model_calls={n_calls} | {time.time() - t0:.1f}s"
        )


def run_approx(
    args: argparse.Namespace,
    model,
    dataset,
    data_ids,
    approx_dir: Path,
    device,
    molecule_ids: list[int] | None = None,
) -> None:
    pairing = resolve_pairing(args.config_approximators)
    for id_explain, data_id in enumerate(data_ids):
        game = build_game(model, dataset, data_id, args.n_layers, device)
        if molecule_ids is not None:
            game_id = f"{args.dataset}_n{args.n_players}"
        else:
            game_id = f"{args.model_type}_{args.dataset}_{args.n_layers}_data{data_id}"
        approximators = get_approximators(
            args.approximators,
            game.n_players,
            args.random_state,
            pairing,
            args.index,
            args.order,
        )
        min_b = game.n_players + 1
        max_b = min(2**game.n_players, args.max_budget)
        budget_range = (
            np.ceil(np.logspace(np.log10(min_b), np.log10(max_b), args.n_budget_steps))
            .clip(min_b, max_b)
            .astype(int)
        )
        for approx in approximators:
            if approx.name == "SVARMIQ" and game.n_players > 20:
                print(f"[SKIP] SVARMIQ on {game_id}: n_players={game.n_players} > 20.")
                continue
            if approx.name == "KernelSHAPIQ" and args.order > 3:
                print(f"[SKIP] KernelSHAPIQ on {game_id}: order={args.order} > 3 not supported.")
                continue
            print(
                f"[APPROX] {approx.name} on {game_id} (n_players={game.n_players}) "
                f"budgets {budget_range[0]}..{budget_range[-1]}"
            )
            for budget in budget_range:
                save_path = approx_dir / (
                    f"{game_id}_{args.config_approximators}_{id_explain}"
                    f"_{approx.name}_{budget}_{args.index}_{args.order}.json"
                )
                if save_path.exists() and not args.override:
                    continue
                try:
                    t0 = time.time()
                    iv = approx.approximate(
                        budget=int(budget),
                        game=game,
                        game_id=game_id,
                        id_explain=id_explain,
                    )
                    wall = time.time() - t0
                    runtime_kwargs = {
                        "total_runtime": wall,
                        **{
                            "total_approximation" if k == "total" else k: v
                            for k, v in approx.runtime_last_approximate_run.items()
                        },
                    }
                    iv.save(save_path, **runtime_kwargs)
                    print(f"  budget={budget} | {wall:.2f}s")
                except Exception as exc:
                    print(f"  [ERR] budget={budget}: {exc}")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = resolve_base_path(args)
    approx_dir = base / "approximations" / "graph"
    truth_dir = base / "ground_truth" / "graph"
    approx_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    # Prefer GraphSHAP-IQ's CustomTUDataset when available (supports
    # FluorideCarbonyl, Benzene, AlkaneCarbonyl, Mutagenicity_XAI).
    # Special-case GraphXAI datasets provided inside GraphSHAP-IQ to avoid
    # importing GraphSHAP-IQ's `tu_dataset.py` which uses absolute `shapiq...`
    # imports and can conflict with the workspace `shapiq` package.
    special = {"FluorideCarbonyl", "Benzene", "AlkaneCarbonyl", "Mutagenicity_XAI"}
    if args.dataset in special:
        root_str = str(GRAPHSHAPIQ_ROOT)
        added = False
        try:
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
                added = True
            gx_pkg = importlib.import_module("graphxai_local.datasets")
            if args.dataset == "Mutagenicity_XAI":
                dataset = getattr(gx_pkg, "Mutagenicity")(root=str(TU_DATA_ROOT), seed=args.graph_seed)
            else:
                cls = getattr(gx_pkg, args.dataset)
                dataset = cls(seed=args.graph_seed)
            print(f"[setup] loaded GraphXAI dataset {args.dataset} from {GRAPHSHAPIQ_ROOT}")
        except Exception as exc:
            print(f"[WARN] GraphXAI dataset load failed: {exc}; falling back to TUDataset")
            try:
                dataset = TUDataset(root=str(TU_DATA_ROOT), name=args.dataset)
            except Exception as exc2:
                print(f"[ERR] TUDataset failed to load {args.dataset}: {exc2}")
                raise
        finally:
            if added:
                try:
                    sys.path.remove(root_str)
                except ValueError:
                    pass
    else:
        # Default: use PyG's TUDataset loader
        try:
            dataset = TUDataset(root=str(TU_DATA_ROOT), name=args.dataset)
            print(f"[setup] loading TU dataset {args.dataset} from {TU_DATA_ROOT} via TUDataset")
        except Exception as exc:
            print(f"[ERR] TUDataset failed to load {args.dataset}: {exc}")
            raise
    print(f"[setup] dataset has {len(dataset)} graphs")

    # Support exact n_players sampling: pick `n_runs` graphs with exactly that node count
    molecule_ids = None
    if args.n_players is not None:
        if args.n_runs is None:
            raise SystemExit("--n_players requires --n_runs to be set")
        data_ids = select_graph_indices_exact(
            dataset, n_players=args.n_players, n_runs=args.n_runs, seed=args.graph_seed
        )
        # molecule ids 0..n_runs-1 used in filenames
        molecule_ids = list(range(len(data_ids)))
    else:
        data_ids = select_graph_indices(
            dataset,
            min_players=args.min_players,
            max_players=args.max_players,
            n_graphs=args.n_graphs,
            seed=args.graph_seed,
        )
    print(f"[setup] selected data_ids={data_ids}")
    print(
        "[setup] n_players: "
        + ", ".join(f"{i}->{int(_unwrap_graph(dataset[i]).num_nodes)}" for i in data_ids)
    )

    first_item = _unwrap_graph(dataset[0])
    num_node_features = first_item.num_node_features
    num_classes = int(dataset.num_classes) if hasattr(dataset, "num_classes") else 2

    model, model_id = load_pretrained_gnn(
        args.model_type,
        args.dataset,
        args.n_layers,
        num_node_features,
        num_classes,
        device,
    )
    print(f"[setup] loaded {model_id} on {device}")

    if args.mode == "true":
        run_true(args, model, dataset, data_ids, truth_dir, device, molecule_ids)
    else:
        run_approx(args, model, dataset, data_ids, approx_dir, device, molecule_ids)


if __name__ == "__main__":
    main()
