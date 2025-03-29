#!/bin/bash

#SBATCH --job-name=samia-generate-pareto-fronts-job
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=v100

source movie-rev-env/bin/activate   # activate python venv with all dependencies installed 

python3 finetune_gpt2.py    # generate pareto fronts
