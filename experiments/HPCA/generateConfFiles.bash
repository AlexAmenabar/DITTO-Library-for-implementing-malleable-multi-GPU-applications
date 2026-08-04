#!/bin/bash

# Arrays of parameters
pFixed=(100 75 0)
pMoldable=(0 0 0)
pFlexible=(0 25 100)

pIterative=(33 0)
pPhases=(33 0)
pComm=(33 100)

policy=(0 1 2 3)

# Array to store filenames
filenames=()
sbatch_filenames=()

mkdir -p sbatch

for p in "${policy[@]}"; do

    if [[ $p -eq 0 ]]; then
        prefix="UT"
        executable="./bin/scheduler"
    elif [[ $p -eq 1 ]]; then
        prefix="TP"
        executable="./bin/schedulerTopo"
    elif [[ $p -eq 2 ]]; then
        prefix="EP"
        executable="./bin/schedulerExpand"
    elif [[ $p -eq 3 ]]; then
        prefix="ET"
        executable="./bin/schedulerExpandTopo"
    fi

    for ((i=0; i<${#pFixed[@]}; i++)); do
        for ((j=0; j<${#pIterative[@]}; j++)); do

            jobName="${prefix}_J_${pFixed[$i]}_${pFlexible[$i]}_A_${pIterative[$j]}_${pPhases[$j]}_${pComm[$j]}"

            workload="experiments/HPCA/Workloads/workload_J_${pFixed[$i]}_${pFlexible[$i]}_A_${pIterative[$j]}_${pPhases[$j]}_${pComm[$j]}.txt"

            sbatchFile="experiments/HPCA/WorkloadsSlurm/${jobName}.sbatch"

            cat > "$sbatchFile" <<EOF
#!/bin/bash

#SBATCH --job-name=${jobName}
#SBATCH --output=experiments/HPCA/WorkloadsOutput/${jobName}.out
#SBATCH --error=experiments/HPCA/WorkloadsOutput/${jobName}.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=07-00:00:00
#SBATCH --mem-per-cpu=10GB

#SBATCH --partition=GPU
#SBATCH --gres=gpu:rtxa5000:8

bnd exec --devel nvidia-smi topo -m > topoRaw.txt
bnd run --gpu python3 topo2matrix.py topoRaw.txt topoMatrix.txt

bnd exec --devel ${executable} \
8 \
${workload} \
topoMatrix.txt \
experiments/HPCA/WorkloadsOutput/events_${jobName}.txt \
experiments/HPCA/WorkloadsOutput/monitor_${jobName}.txt \
experiments/HPCA/WorkloadsOutput/out_${jobName}.txt \
10 \
3600
EOF

            chmod +x "$sbatchFile"

        done
    done
done