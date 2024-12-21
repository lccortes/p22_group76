#!/bin/sh 

# custom GPU selector commands
#BSUB -q gpuv100
#BSUB -J Luisa_3
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -B
#BSUB -N
#BSUB -R "span[hosts=1]"

# Load environment variables
source ./.env-vars

# set logfiles to capture stdout and stderr
date=$(date +%Y%m%d_%H%M)
mkdir -p "${REPO}/job_out/${date}"
touch "${REPO}/job_out/${date}/gpu.out"
touch "${REPO}/job_out/${date}/gpu.err"
#BSUB -o ${REPO}/job_out/${date}/gpu.out
#BSUB -e ${REPO}/job_out/${date}/gpu.err

# check GPU information in logs (can be useful for debugging)
nvidia-smi

# Activate venv created by uv
source ${REPO}/.venv/bin/activate

# run training
python3 CNN_baseline_py/main.py
