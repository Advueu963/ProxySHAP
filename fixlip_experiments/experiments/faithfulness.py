import argparse

parser = argparse.ArgumentParser(description="main")
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument(
    "--path_input", type=str, default="fixlip_experiments/results/mscoco"
)
parser.add_argument("--path_output", type=str, default="fixlip_experiments/results/")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=10)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--random_state", default=0, type=int)
parser.add_argument("--is_cross_modal", action="store_true")
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
START = args.start
STOP = args.stop
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state

N_EVAL_COALITIONS = 1000  # number of coalitions to evaluate the faithfulness metrics on

# print the settings
print(f"MODEL_NAME: {MODEL_NAME}", flush=True)
print(f"PATH_INPUT: {PATH_INPUT}", flush=True)
print(f"PATH_OUTPUT: {PATH_OUTPUT}", flush=True)
print(f"START: {START}", flush=True)
print(f"STOP: {STOP}", flush=True)
print(f"BATCH_SIZE: {BATCH_SIZE}", flush=True)
print(f"RANDOM_STATE: {RANDOM_STATE}", flush=True)
print(f"N_EVAL_COALITIONS: {N_EVAL_COALITIONS}", flush=True)
import sys
import os
import time

# import clip
import torch

torch.set_float32_matmul_precision("high")
from transformers import CLIPProcessor, CLIPModel
import datasets
from shapiq import InteractionValues
import pandas as pd

sys.path.append("../")
import src
from src.sampler import CoalitionSampler
import numpy as np
import scipy as sp
start_time = time.time()


def load_metadata(path: str) -> dict:
    import json

    with open(path, "r") as f:
        json_data = f.read()
    try:
        meta_data = json.loads(json_data).get("parameters", {}).get("kwargs", {})
    except Exception:
        print(f"Error loading runtime for {path}")
    return meta_data

if __name__ == "__main__":
    # with wandb.init(project="", name=f'{PATH_OUTPUT}/{MODEL_NAME}/faith', config=args) as run:

    RESULT_DATA: list[dict[str, float]] = []

    df_metadata = pd.read_csv(
        os.path.join(PATH_OUTPUT, MODEL_NAME, "mscoco_predictions.csv"), index_col=0
    )
    top_ids = (
        df_metadata.sort_values("logit", ascending=False).iloc[START:STOP, :].index
    )

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions", split="test", streaming=True
    )

    model_huggingface = CLIPModel.from_pretrained(MODEL_NAME)
    model_huggingface.to("cuda" if torch.cuda.is_available() else "cpu")
    processor_huggingface = CLIPProcessor.from_pretrained(MODEL_NAME)

    # model_openai, processor_openai = clip.load("ViT-B/32" if MODEL_NAME.endswith("32") else "ViT-B/16", device=1)

    n_iter = 0

    sample_p = 0.5
    is_cross_modal = args.is_cross_modal #True if MODEL_NAME.endswith("16") else False
    for i, d in enumerate(dataset):
        ground_truth_values = None
        sampling_size_weights_banzhaf = None
        sampling_size_weights_shapley = None
        coalition_matrix = None
        if i not in top_ids:
                continue
        n_iter += 1
        for budget in [100, 200, 500, 1000, 2000, 5000, 10000, 15_000]:
            # load the interaction values only if all are present --------------------------------------
            explanations_huggingface = {}
            metadata_huggingface = {}
            explanations_openai = {}

            # # banzhaf 0.3  -------------------------------------------------------------------------
            # banzhaf_p = "0.3"
            # interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order1_{i}.pkl")
            # banzhaf_1_03 = InteractionValues.load(interaction_path)
            # explanations_huggingface["banzhaf_1_03"] = banzhaf_1_03
            # interaction_path = os.path.join(PATH_INPUT, MODEL_NAME, "banzhaf", banzhaf_p, f"iv_order2_{i}.pkl")
            # banzhaf_2_03 = InteractionValues.load(interaction_path)
            # explanations_huggingface["banzhaf_2_03"] = banzhaf_2_03

            # banzhaf 0.5  -------------------------------------------------------------------------
            banzhaf_p = "0.5"
            base_path = os.path.join(PATH_INPUT, MODEL_NAME)
            
            if is_cross_modal:
                regression_o1_name = f"iv_order1_{i}_{budget}_regression_crossmodal.json"
                regression_o2_name = f"iv_order2_{i}_{budget}_regression_crossmodal.json"
                proxyshap_o1_name = f"iv_order1_{i}_{budget}_proxyshap_crossmodal.json"
                proxyshap_o2_name = f"iv_order2_{i}_{budget}_proxyshap_crossmodal.json"
                proxyshapna_o1_name = f"iv_order1_{i}_{budget}_proxyshap-noadjustment_crossmodal.json"
                proxyshapna_o2_name = f"iv_order2_{i}_{budget}_proxyshap-noadjustment_crossmodal.json"
                proxyspex_o1_name = f"iv_order1_{i}_{budget}_proxyspex_crossmodal.json"
                proxyspex_o2_name = f"iv_order2_{i}_{budget}_proxyspex_crossmodal.json"
            else:
                regression_o1_name = f"iv_order1_{i}_{budget}_regression_144.json"
                regression_o2_name = f"iv_order2_{i}_{budget}_regression_144.json"
                proxyshap_o1_name = f"iv_order1_{i}_{budget}_proxyshap_144.json"
                proxyshap_o2_name = f"iv_order2_{i}_{budget}_proxyshap_144.json"
                proxyshapna_o1_name = f"iv_order1_{i}_{budget}_proxyshap-noadjustment_144.json"
                proxyshapna_o2_name = f"iv_order2_{i}_{budget}_proxyshap-noadjustment_144.json"
                proxyspex_o1_name = f"iv_order1_{i}_{budget}_proxyspex_144.json"
                proxyspex_o2_name = f"iv_order2_{i}_{budget}_proxyspex_144.json"
            
            
            interaction_path = os.path.join(
                base_path,
                regression_o1_name,
            )
            try:
                banzhaf_1_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_1_05_regression"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_1_05_regression"] = banzhaf_1_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            interaction_path = os.path.join(
                base_path,
                regression_o2_name,
            )
            try:
                banzhaf_2_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_2_05_regression"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_2_05_regression"] = banzhaf_2_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyshap_o1_name,
            )
            try:
                banzhaf_1_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_1_05_proxyshap"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_1_05_proxyshap"] = banzhaf_1_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyshap_o2_name,
            )
            try:
                banzhaf_2_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_2_05_proxyshap"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_2_05_proxyshap"] = banzhaf_2_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyshapna_o1_name,
            )
            try:
                banzhaf_1_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_1_05_proxyshap-noadjustment"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_1_05_proxyshap-noadjustment"] = banzhaf_1_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyshapna_o2_name,
            )
            try:
                banzhaf_2_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_2_05_proxyshap-noadjustment"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_2_05_proxyshap-noadjustment"] = banzhaf_2_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
                
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyspex_o1_name,
            )
            try:
                banzhaf_1_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_1_05_proxyspex"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_1_05_proxyspex"] = banzhaf_1_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
            interaction_path = os.path.join(
                base_path,
                # MODEL_NAME,
                # "banzhaf",
                # banzhaf_p,
                proxyspex_o2_name,
            )
            try:
                banzhaf_2_05 = InteractionValues.load(interaction_path)
                metadata_huggingface["banzhaf_2_05_proxyspex"] = load_metadata(
                    interaction_path
                )
                explanations_huggingface["banzhaf_2_05_proxyspex"] = banzhaf_2_05
            except Exception:
                print(f"Could not load {interaction_path}, skipping instance.")
           

            # load image/text and create games ----------------------------------------------------------
            print(f"budget {budget}, iter: {n_iter}/{STOP - START}", flush=True)
            input_image = d["jpg"]
            input_text = d["txt"].split("\n")[df_metadata.loc[i, "best_text_id"].item()]
            game_huggingface = src.game_huggingface.VisionLanguageGame(
                model_huggingface,
                processor_huggingface,
                input_image=input_image,
                input_text=input_text,
                batch_size=BATCH_SIZE,
            )
            
            ## Pre-compute the gt_values for the sampled coalitions to speed up evaluation
            if ground_truth_values is None:
                sampling_size_weights_banzhaf = np.array([
                    sp.special.binom(game_huggingface.n_players, k) * (sample_p ** k) * (
                    (1 - sample_p) ** (game_huggingface.n_players - k)) for k in range(game_huggingface.n_players + 1)
                ])
                sampling_size_weights_shapley = np.zeros(game_huggingface.n_players + 1)
                for coalition_size in range(1, game_huggingface.n_players):
                    sampling_size_weights_shapley[coalition_size] = 1 / (coalition_size * (game_huggingface.n_players - coalition_size))
                # Sample coalitions (sample mode is always banzhaf for computing gt values)
                sampling_size_weights = sampling_size_weights_banzhaf
                enforce_empty_full = False

                sampler = CoalitionSampler(
                    n_players=game_huggingface.n_players,
                    sampling_weights=sampling_size_weights,
                    enforce_empty_full=enforce_empty_full,
                    pairing_trick=False,
                    random_state=i
                )
                sampler.sample(N_EVAL_COALITIONS)
                coalition_matrix = sampler.coalitions_matrix

                # get ground-truth values
                ground_truth_values = game_huggingface.value_function(coalition_matrix)
                empty_prediction = game_huggingface.normalization_value
                ground_truth_values -= empty_prediction
    
            
            # game_openai = src.game_openai.CLIPGame(
            #     model_openai, processor_openai,
            #     input_image=input_image,
            #     input_text=input_text,
            #     batch_size=BATCH_SIZE,
            #     patch_size=32 if MODEL_NAME.endswith("32") else 16
            # )

            # compute the faithfulness metrics for different p -----------------------------------------
            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_huggingface,
            #     explanations=explanations_huggingface,
            #     sample_p=0.3,
            #     sample_mode="banzhaf",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)
            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_openai,
            #     explanations=explanations_openai,
            #     sample_p=0.3,
            #     sample_mode="banzhaf",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)

            results = src.evaluation.eval_faithfulness_one_game_given_values(
                ground_truth_values=ground_truth_values,
                coalition_matrix=coalition_matrix,
                sampling_size_weights_banzhaf=sampling_size_weights_banzhaf,
                sampling_size_weights_shapley=sampling_size_weights_shapley,
                explanations=explanations_huggingface,
                sample_p=sample_p,
                sample_mode="banzhaf",
                n_eval_coalitions=N_EVAL_COALITIONS,
                instance_id=i,
            )
            ### ADD RUNTIMES TO RESULTS ------------------------------------------------
            for res in results:
                method_name = res["method_name"]
                if method_name in metadata_huggingface:
                    res.update(metadata_huggingface[method_name])
                    res.update({"budget": budget})
            RESULT_DATA.extend(results)  
            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_openai,
            #     explanations=explanations_openai,
            #     sample_p=0.5,
            #     sample_mode="banzhaf",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)

            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_huggingface,
            #     explanations=explanations_huggingface,
            #     sample_p=0.7,
            #     sample_mode="banzhaf",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)
            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_openai,
            #     explanations=explanations_openai,
            #     sample_p=0.7,
            #     sample_mode="banzhaf",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)

            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_huggingface,
            #     explanations=explanations_huggingface,
            #     sample_mode="shapley",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)
            # results = src.evaluation.eval_faithfulness_one_game(
            #     game=game_openai,
            #     explanations=explanations_openai,
            #     sample_mode="shapley",
            #     n_eval_coalitions=N_EVAL_COALITIONS,
            #     instance_id=i
            # )
            # RESULT_DATA.extend(results)

            if n_iter == 1 or n_iter % 5 == 0:
                # store the current results by overwriting a temporary file
                df_results = pd.DataFrame(RESULT_DATA)
                df_results.to_csv(
                    os.path.join(
                        PATH_OUTPUT,
                        MODEL_NAME,
                        f"eval_faithfulness_temp_{N_EVAL_COALITIONS}_{START}_{STOP}.csv",
                    ),
                    index=False,
                )

    # save final results ---------------------------------------------------------------------------
    if is_cross_modal:
        save_name= f"eval_faithfulness_crossmodal_{N_EVAL_COALITIONS}_{START}_{STOP}.csv"
    else:
        save_name= f"eval_faithfulness_{N_EVAL_COALITIONS}_{START}_{STOP}_144.csv"
    df_results = pd.DataFrame(RESULT_DATA)
    df_results.to_csv(
        os.path.join(
            PATH_OUTPUT,
            MODEL_NAME,
            save_name,
        ),
        index=False,
    )

# print the time taken -------------------------------------------------------------------------
elapsed_time = time.time() - start_time
print(f"Time taken: {elapsed_time:.2f} seconds")
