from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
from shapiq.approximator.sampling import CoalitionSampler
from shapiq_benchmark.load import BenchmarkFactory
from shapiq_benchmark.tabpfn import TabPFNBenchmark


if __name__ == "__main__":
    """
    This script runs selected approximation algorithms on explanation games that use baseline
    imputatation, which were pre-computed in the shapiq library. The ground truth values
    are computed using exhaustive evaluation. Approximations are stored in
    /approximations/exhaustive/ and ground truth values in /ground_truth/exhaustive/.
    """
    RANDOM_STATE = 40  # random state for the games
    # ID_CONFIG_APPROXIMATORS = 40  # PAIRING=False, REPLACEMENT=True
    # ID_CONFIG_APPROXIMATORS = 39  # PAIRING=False, REPLACEMENT=False
    # ID_CONFIG_APPROXIMATORS = 38  # PAIRING=True, REPLACEMENT=True

    # BENCHMARKS = BenchmarkFactory.create_benchmarks_interactive(
    #     game_config_names=[
    #         "BikeSharingLocalXAI",
    #         "CaliforniaHousingLocalXAI",
    #         "AdultCensusLocalXAI",
    #         "CaliforniaHousingLocalXAI",
    #         "ForestFiresLocalXAI",
    #         "RealEstateLocalXAI",
    #     ],
    #     n_games=30 ,
    #     approximation_methods=[
    #         "SVARMIQ",
    #         "SHAPIQ",
    #         "KernelSHAPIQ",
    #         "RegressionMSRIQ",
    #         "RegressionMSRIQ-NoAdjustment",
    #         "Linear-RECAP",
    #         "Tree-RECAP",
    #         "ProxySpex",
    #         "Linear-NoAdjustment"
    #     ],
    #     index="SV",
    #     order=1,
    #     config_path="shapiq-benchmark/configurations_tabpfn/",
    #     config_save_name="configuration_tabpfn",
    # )
    BENCHMARKS = BenchmarkFactory.load_benchmarks_from_json(
        config_path="shapiq-benchmark/benchmarks/configuration_tabpfn.json"
    )
    # N_DATASETS = len(BENCHMARKS)
    N_ITERATIONS = 30
    # Subsample the approx list if we have a SLURM_ARRAY_TASK_ID
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = (int(os.environ["SLURM_ARRAY_TASK_ID"]) // N_ITERATIONS)
        BENCHMARKS = {k: v for i, (k, v) in enumerate(BENCHMARKS.items()) if i == task_id}
        print(
            "Subsampling benchmark list to only run:", list(BENCHMARKS.keys())[0]
        )
    for game_identifier, benchmark_info in BENCHMARKS.items():
        games = benchmark_info["games"]
        games_enumerated = list(enumerate(games))
        for id_explain, game_instance in games_enumerated:
            x_explain = game_instance.x.astype(np.float32)
            print("Considering game:", game_identifier, "explanation id:", id_explain, len(games_enumerated))
            slurm_id_explain = int(os.environ["SLURM_ARRAY_TASK_ID"]) % N_ITERATIONS if "SLURM_ARRAY_TASK_ID" in os.environ else None
            if (slurm_id_explain is None) or (id_explain == slurm_id_explain):
                print("Running precomputation")
                game_instance.precompute()
                print("Precomputation done")
                
                if not os.path.exists(
                    "shapiq-benchmark/src/shapiq_benchmark/precomputed/" + game_identifier
                ):
                    os.makedirs(
                        "shapiq-benchmark/src/shapiq_benchmark/precomputed/" + game_identifier
                    )
                save_path = Path(
                    "shapiq-benchmark/src/shapiq_benchmark/precomputed/"
                    + game_identifier + "/"
                    +"model_name=tabpfn_imputer=tabpfn_"
                    + str(id_explain)
                    + ".json"
                )
                game_instance.save(save_path)
                print(f"Precomputed game values saved to {save_path}")