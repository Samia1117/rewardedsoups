#!/bin/bash

#SBATCH --job-name=samia-generate-pareto-fronts-job
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000

source movie-rev-env/bin/activate   # activate python venv with all dependencies installed 

python3 check_gpu_usage.py

python3 finetune_gpt2.py