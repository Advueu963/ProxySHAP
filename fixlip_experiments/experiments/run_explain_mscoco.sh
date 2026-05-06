#!/bin/bash

#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --partition=lrz-v100x2 #lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=00:30:00
#SBATCH --container-image=/dss/dsshome1/0D/ra98xir2/enroot/nvidia+pytorch+25.11-py3.sqsh # (or 'nvcr.io/nvidia/pytorch:23.10-py3')
#SBATCH --container-mounts=/dss/dsshome1/0D/ra98xir2/msr_int_iq/:/workspace,/dss/dssfs02/lwp-dss-0001/pn49je/pn49je-dss-0000/ra98xir2:/workspace/save       # Mount host project directory into /workspace

set -e
hostname; pwd; date

source .venv/bin/activate

date

cd fixlip_experiments/experiments/

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python explain_mscoco.py --model_name $1 \
                         --path_input $2 \
                         --path_output $3 \
                         --start $4 \
                         --stop $5 \
                         --mode $6 \
                         --p_sampler $7 \
                         --batch_size $8 \
                         --budget $9 \
                         --random_state ${10} \
                         --approximation_type ${11}

date
