#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --time=23:00:00
#SBATCH --account=plgcredibleai2025-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=slurm_logs/%A.log
#SBATCH --job-name=metagame

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# IMPORTANT: load the modules for machine learning tasks and libraries
module load GCCcore/13.2.0
module load binutils/2.40
module add ML-bundle/24.06a
ml ML-bundle/24.06a

# activate virtual environment
source .venv/bin/activate

# check python version and location
which python
python -V

# change ssl certificate file
export SSL_CERT_FILE=$(python -m certifi)

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