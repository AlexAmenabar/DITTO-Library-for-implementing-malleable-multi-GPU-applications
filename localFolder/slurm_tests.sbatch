#!/bin/bash

# Job name
#SBATCH --job-name=DITTO

# Define the files which will contain the Standard and Error output
#SBATCH --output=a.out
#SBATCH --error=a.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
#SBATCH --cpus-per-task=1

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=10GB

# Select one partition
## CPU
## GPU
#SBATCH --partition=TEST
#SBATCH --gres=gpu:rtx2080ti:4
##SBATCH --gres=gpu:1

bnd exec --gpu ./toy_app

