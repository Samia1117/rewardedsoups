#!/bin/bash

#SBATCH --job-name=samia-gtp2-ft-job
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=v100

source movie-rev-env/bin/activate   # activate python venv with all dependencies installed 

python3 finetune_gpt2.py    # finetune gpt2 with mixed reward during PPO training
