#!/bin/bash

#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=hlai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=end
#SBATCH --mail-user=S.Thies@campus.lmu.de
#SBATCH --export=NONE
#SBATCH --time=06:00:00

set -e
hostname; pwd; date

source ../../.venv/bin/activate

date

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
