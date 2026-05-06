#!/bin/bash
#SBATCH -J faithfulness
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=hlai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --dependency=afterok:67747 

set -e
hostname; pwd; date


source ../../.venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


date

# python faithfulness.py \
#     --model_name openai/clip-vit-base-patch32 \
#     --path_input "../../storage/ra98xir2/ra98xir2/fixlip_experiments/results/explanations/mscoco" \
#     --path_output ../results \
#     --start 0 \
#     --stop 30
# date
# echo "-----------------------------------"
python faithfulness.py \
    --model_name openai/clip-vit-base-patch16 \
    --path_input "../../storage/ra98xir2/ra98xir2/fixlip_experiments/results/explanations/mscoco" \
    --path_output ../results \
    --start 0 \
    --stop 30 


date
