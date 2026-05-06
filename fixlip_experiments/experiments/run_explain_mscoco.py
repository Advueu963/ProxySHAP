import os
import subprocess

PATH_INPUT = "../results"
PATH_OUTPUT = "/workspace/save/ra98xir2/fixlip_experiments/results/explanations/mscoco"
RANDOM_STATE = 0

START = 0
STOP = 30

for model_name, batch_size in {
    #"openai/clip-vit-base-patch32": 64,
    "openai/clip-vit-base-patch16": 64,
}.items():
    for mode in [
        'banzhaf', 
    ]:
        if mode == "banzhaf":
            for approximation_type in [
                #"regression",
                #"proxyshap",
                #"proxyshap-noadjustment",
                "proxyshap-default",
                #"proxyshap-hpo",
                #"proxyspex",
                #"surrogate",
            ]:
                for budget in [100,200,500,1000,2000,5000,10000]:
                    # for batches in range(1):
                    #     START = batches * 20
                    #     STOP = (batches + 1) * 20
                        for p_sampler in [
                            0.5,
                        ]:
                            subprocess.run([
                                "sbatch", 
                                "-J", 
                                f"expl_mscoco_{model_name.replace('/', '_')}_{mode}_p{p_sampler}_start{START}_stop{STOP}",
                                "run_explain_mscoco.sh", 
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
        elif mode == "shapley":
            # empirical adjustment of budget to equalize computation time
            budget = 2**17
            subprocess.run([
                "sbatch", 
                "run_explain_mscoco.sh", 
                model_name,
                os.path.join(PATH_INPUT, model_name),
                os.path.join(PATH_OUTPUT, model_name, mode),
                str(START),
                str(STOP),
                mode,
                str(0.5),
                str(budget),
                str(batch_size),
                str(RANDOM_STATE)
            ])