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
# module load GCCcore/13.2.0
# module load binutils/2.40
module add ML-bundle/24.06a
ml ML-bundle/24.06a

# activate virtual environment
PATH_ENV=$PLG_GROUPS_STORAGE/plggmi2ai/plghbaniecki/msr_int_iq
cd $PATH_ENV
uname -m
uv sync
source .venv/bin/activate

# run exp
PATH_REPO=/$PLG_GROUPS_STORAGE/plggmi2ai/plghbaniecki/msr_int_iq/fixlip_experiments/experiments
cd $PATH_REPO

# check python version and location
which python
python -V

# change ssl certificate file
export SSL_CERT_FILE=$(python -m certifi)

python test.py

date