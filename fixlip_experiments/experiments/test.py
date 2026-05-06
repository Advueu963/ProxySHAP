import argparse
from datetime import time
from pathlib import Path
import time

parser = argparse.ArgumentParser(description="main")
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--path_input", type=str, default="../results")
parser.add_argument(
    "--path_output", type=str, default="/net/storage/pr3/plgrid/plggmi2ai/plghbaniecki/msr_int_iq_results/explanations/mscoco/openai/clip-vit-base-patch16"
)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=30)
parser.add_argument("--mode", type=str, default="banzhaf")
parser.add_argument("--p_sampler", default=0.5, type=float)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--random_state", default=0, type=int)
parser.add_argument("--approximation_type", type=str, default="proxyshap-noadjustment")
parser.add_argument("--budget", type=int, default=1000)
args = parser.parse_args()
MODEL_NAME = args.model_name
PATH_INPUT = args.path_input
PATH_OUTPUT = args.path_output
START = args.start
STOP = args.stop
MODE = args.mode
P_SAMPLER = args.p_sampler
BATCH_SIZE = args.batch_size
RANDOM_STATE = args.random_state
BUDGET = args.budget
APPROXIMATION_TYPE = args.approximation_type

print(f"-- Input: MS COCO", flush=True)
print(f"-- Output: {PATH_OUTPUT}", flush=True)
print(f"-- Model: {MODEL_NAME}", flush=True)
print(f"-- Mode: {MODE}", flush=True)
print(f"-- P sampler: {P_SAMPLER}", flush=True)
print(f"-- Approximation type: {APPROXIMATION_TYPE}", flush=True)


import torch

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"- Device: {DEVICE}", flush=True)
torch.set_float32_matmul_precision("high")

from transformers import CLIPProcessor, CLIPModel
import datasets
import pandas as pd

import os

# if not os.path.exists(PATH_OUTPUT):
#     os.makedirs(PATH_OUTPUT)

import sys

sys.path.append("../")
import src

src.utils.set_seed(RANDOM_STATE)

import wandb
import matplotlib.pyplot as plt

with wandb.init(project="", name=f'{PATH_OUTPUT}/test', config=args) as run:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    dataset = datasets.load_dataset(
        "clip-benchmark/wds_mscoco_captions", split="test", streaming=True
    )

    size_clique = 72
    run.config.update({"size_clique": size_clique, "start": START, "stop": STOP})

    df_metadata = pd.read_csv(
        os.path.join(PATH_INPUT, MODEL_NAME, "mscoco_predictions.csv"), index_col=0
    )

    wandb.finish()