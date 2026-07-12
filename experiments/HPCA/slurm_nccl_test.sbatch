#!/bin/bash

# Job name
#SBATCH --job-name=DITTO_NCCL

# Define the files which will contain the Standard and Error output
#SBATCH --output=experiments/HPCA/DITTO_NCCL_2.out
#SBATCH --error=experiments/HPCA/DITTO_NCCL_2.err

# Number of tasks that compose the job
#SBATCH --ntasks=1


# Advanced use
#SBATCH --cpus-per-task=8

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=07-00:00:00 

# Select one partition
## CPU
## GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtxa5000:8
##SBATCH --gres=gpu:rtx3090:4
##SBATCH --gres=gpu:rtx2080ti:4
##SBATCH --gres=gpu:1

## create topology
#export NCCL_ALGO=Tree
#export NCCL_DEBUG=INFO


bnd run --gpu python3 topo2matrix.py topoRaw.txt topoMatrix.txt

bnd exec --devel ./bin/ncclTest 8 experiments/communicationsExperiments/workload.txt topoMatrix.txt  commEventsNCCL.txt commUsageNCCL.txt commOutputNCCL.txt 

#rm -f topoMatrix.txt