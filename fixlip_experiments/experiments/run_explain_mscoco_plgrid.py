import os
import subprocess

PATH_INPUT = "../results"
PATH_OUTPUT = "/net/storage/pr3/plgrid/plggmi2ai/plghbaniecki/msr_int_iq_results/explanations/mscoco"
RANDOM_STATE = 0

START = 0
STOP = 30

for model_name, batch_size in {
    # "openai/clip-vit-base-patch32": 64,
    "openai/clip-vit-base-patch16": 64,
}.items():
    for mode in [
        'banzhaf', 
    ]:
        if mode == "banzhaf":
            for approximation_type in [
                "regression",
                # "proxyshap",
                # "proxyshap-noadjustment",
                # "proxyspex",
            ]:
                for budget in [
                        # 200, 
                        # 500, 
                        # 1000, 
                        # 2000, 
                        # 5000, 
                        10000
                    ]:
                        for p_sampler in [
                            0.5,
                        ]:
                            subprocess.run([
                                "sbatch", 
                                "run_explain_mscoco_plgrid.sh", 
                                model_name,
                                os.path.join(PATH_INPUT),
                                os.path.join(PATH_OUTPUT, model_name),
                                str(START),
                                str(STOP),
                                mode,
                                str(p_sampler),
                                str(batch_size),
                                str(budget),
                                str(RANDOM_STATE),
                                str(approximation_type)
                            ])