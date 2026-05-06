#!/bin/bash
#SBATCH -J faithfulness
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --container-image=/dss/dsshome1/0D/ra98xir2/enroot/nvidia+pytorch+25.11-py3.sqsh # (or 'nvcr.io/nvidia/pytorch:23.10-py3')
#SBATCH --container-mounts=/dss/dsshome1/0D/ra98xir2/msr_int_iq/:/workspace,/dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2:/workspace/save       # Mount host project directory into /workspace
set -e
hostname; pwd; date


source .venv/bin/activate

cd fixlip_experiments/experiments/
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


date

# python faithfulness.py \
#     --model_name openai/clip-vit-base-patch32 \
#     --path_input /workspace/save/ra98xir2/fixlip_experiments/results/explanations/mscoco \
#     --path_output ../results \
#     --start 0 \
#     --stop 30 \
#     --is_cross_modal
# date
# echo "-----------------------------------"
python faithfulness.py \
    --model_name openai/clip-vit-base-patch16 \
    --path_input /workspace/save/ra98xir2/fixlip_experiments/results/explanations/mscoco \
    --path_output ../results \
    --start 0 \
    --stop 30 #\
    #--is_cross_modal

date
