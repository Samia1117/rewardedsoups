#!/bin/bash

#SBATCH --job-name=caroline-gtp2-ft-job
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=a6000

source movie-rev-env/bin/activate   # activate python venv with all dependencies installed 

python3 generate_sample_reviews.py