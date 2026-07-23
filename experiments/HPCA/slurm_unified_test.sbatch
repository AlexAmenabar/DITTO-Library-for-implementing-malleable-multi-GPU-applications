#!/bin/bash

# Job name
#SBATCH --job-name=DITTO_UNIFIED

# Define the files which will contain the Standard and Error output
#SBATCH --output=experiments/HPCA/DITTO_UNIFIED.out
#SBATCH --error=experiments/HPCA/DITTO_UNIFIED.err

# Number of tasks that compose the job
#SBATCH --ntasks=1


# Advanced use
#SBATCH --cpus-per-task=16
#SBATCH --time=07-00:00:00 

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=20GB

# Select one partition
## CPU
## GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtxa5000:8
##SBATCH --gres=gpu:rtx3090:4
##SBATCH --gres=gpu:rtx2080ti:4
##SBATCH --gres=gpu:1

bnd exec --devel nvidia-smi topo -m > topoRaw.txt

bnd run --gpu python3 topo2matrix.py topoRaw.txt topoMatrix.txt

bnd exec --devel ./bin/unifiedTest 8 experiments/communicationsExperiments/workload.txt topoMatrix.txt commEventsNCCL.txt commUsageNCCL.txt commOutputNCCL.txt 
#valgrind --leak-check=full
#compute-sanitizer --tool memcheck