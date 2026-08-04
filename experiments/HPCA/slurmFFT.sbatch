#!/bin/bash

# Job name
#SBATCH --job-name=FFT

# Define the files which will contain the Standard and Error output
#SBATCH --output=experiments/HPCA/FFT.out
#SBATCH --error=experiments/HPCA/FFT.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

#SBATCH --time=7-00:00:00 

# Advanced use
#SBATCH --cpus-per-task=32

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=10GB

# Select one partition
## CPU
## GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtxa5000:8
##SBATCH --gres=gpu:rtx3090:4
##SBATCH --gres=gpu:rtx2080ti:4
##SBATCH --gres=gpu:1
 
bnd exec --devel ./bin/fft 8 topoMatrixTest.txt experiments/communicationsExperiments/workload.txt commEventsTimes.txt commUsageTimes.txt commOutputTimes.txt 
#compute-sanitizer --tool memcheck 