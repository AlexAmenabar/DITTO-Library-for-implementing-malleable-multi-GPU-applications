#!/bin/bash

# Job name
#SBATCH --job-name=DITTO_Reconfs

# Define the files which will contain the Standard and Error output
#SBATCH --output=experiments/HPCA/DITTO_ReconfsTest.out
#SBATCH --error=experiments/HPCA/DITTO_ReconfsTest.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

#SBATCH --time=01:00:00 

# Advanced use
#SBATCH --cpus-per-task=16

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=4GB

# Select one partition
## CPU
## GPU
#SBATCH --partition=TEST
##SBATCH --gres=gpu:rtxa5000:8
##SBATCH --gres=gpu:rtx3090:4
#SBATCH --gres=gpu:rtx2080ti:4
##SBATCH --gres=gpu:1
 
bnd exec --devel ./bin/reconfs 4 topoMatrixTest.txt experiments/communicationsExperiments/workload.txt commEvents.txt commUsage.txt commOutput.txt 