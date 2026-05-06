from shapiq_benchmark.configuration import GameConfigurator, BENCHMARK_CONFIGURATIONS, GAME_CLASS_TO_NAME_MAPPING
from shapiq_games.synthetic import SOUM
from shapiq_games.benchmark import (
    AdultCensusDataValuation,
    BikeSharingDataValuation,
    CaliforniaHousingDataValuation,
    ImageClassifierLocalXAI,
)

if __name__ == "__main__":
    for game_class, game_config in BENCHMARK_CONFIGURATIONS.items():
        idx_configs = 0
        for configs in game_config:
            config_path = ""
            if len(game_config) > 1:
                ## We are in the ImageLocalXAI Case
                if (game_class.__qualname__ == ImageClassifierLocalXAI.__qualname__):
                    image_models = ["ResNet18w14Superpixel", "ViT3by3Patches","ViT4by4Patches"]
                    config_path = "shapiq-benchmark/configurations_exhaustive/{}.json".format(
                        #GAME_CLASS_TO_NAME_MAPPING[game_class],
                        image_models[idx_configs]
                    )
                ## We are in the Case of AdultCenssusDataeValuation with different datasets
                elif (game_class.__qualname__ == AdultCensusDataValuation.__qualname__):
                    n_players = ["10_points","14_points"]
                    config_path = "shapiq-benchmark/configurations_exhaustive/{}_{}.json".format(
                        GAME_CLASS_TO_NAME_MAPPING[game_class],
                        n_players[idx_configs]
                    )
                elif (game_class.__qualname__ == BikeSharingDataValuation.__qualname__):
                    n_players = ["10_points","14_points"]
                    config_path = "shapiq-benchmark/configurations_exhaustive/{}_{}.json".format(
                        GAME_CLASS_TO_NAME_MAPPING[game_class],
                        n_players[idx_configs]
                    )
                elif (game_class.__qualname__ == CaliforniaHousingDataValuation.__qualname__):
                    n_players = ["10_points","14_points"]
                    config_path = "shapiq-benchmark/configurations_exhaustive/{}_{}.json".format(
                        GAME_CLASS_TO_NAME_MAPPING[game_class],
                        n_players[idx_configs]
                    )
                ## We are in the case of SOUM
                elif (game_class.__qualname__ == SOUM.__qualname__):
                    soum_types = ["15_players","30_players","50_players"]
                    config_path = "shapiq-benchmark/configurations_exhaustive/{}_{}.json".format(
                        GAME_CLASS_TO_NAME_MAPPING[game_class],
                        soum_types[idx_configs]
                    )
            else:
                config_path = "shapiq-benchmark/configurations_exhaustive/{}.json".format(GAME_CLASS_TO_NAME_MAPPING[game_class])
                
            ## Transform iteration_parameter_values_names to strings
            if "iteration_parameter_values_names" in configs:
                configs["iteration_parameter_values_names"] = [str(name) for name in configs["iteration_parameter_values_names"]]
            GameConfigurator(game_class_or_name=game_class, 
                             config_path=config_path,
                             configurations=configs["configurations"],
                             iteration_parameter=configs.get("iteration_parameter", None),
                             iteration_parameter_values_names=configs.get("iteration_parameter_values_names", list(range(1, 31))),
                             iteration_parameter_values=configs.get("iteration_parameter_values", list(range(1, 31))),
                             n_players=configs.get("n_players", None),
                             precompute=configs.get("precompute", None)
                             ).to_json()
            print(f"Created configuration file at: {config_path}")
            idx_configs += 1
        
