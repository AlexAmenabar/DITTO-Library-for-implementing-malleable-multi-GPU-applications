#!/bin/bash

# Arrays of parameters
pFixed=(100 75 0)
pMoldable=(0 0 0)
pFlexible=(0 25 100)

pIterative=(33 20)
pPhases=(33 10)
pComm=(33 70)

policy=(0 1)

# Array to store filenames
filenames=()
sbatch_filenames=()

mkdir -p sbatch

for p in "${policy[@]}"; do

    if [[ $p -eq 0 ]]; then
        prefix="UT"
        executable="./bin/scheduler"
    else
        prefix="TP"
        executable="./bin/schedulerTopo"
    fi

    for ((i=0; i<${#pFixed[@]}; i++)); do
        for ((j=0; j<${#pIterative[@]}; j++)); do

            jobName="${prefix}_J_${pFixed[$i]}_${pMoldable[$i]}_0_${pFlexible[$i]}_A_${pIterative[$j]}_${pPhases[$j]}_${pComm[$j]}"

            workload="experiments/HPCA/WorkloadsReduced/workload_J_${pFixed[$i]}_${pMoldable[$i]}_0_${pFlexible[$i]}_A_${pIterative[$j]}_${pPhases[$j]}_${pComm[$j]}.txt"

            sbatchFile="experiments/HPCA/WorkloadsSlurmReduced/${jobName}.sbatch"

            cat > "$sbatchFile" <<EOF
#!/bin/bash

#SBATCH --job-name=${jobName}
#SBATCH --output=experiments/HPCA/WorkloadsOutputReduced/${jobName}.out
#SBATCH --error=experiments/HPCA/WorkloadsOutputReduced/${jobName}.err

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
experiments/HPCA/WorkloadsOutputReduced/events_${jobName}.txt \
experiments/HPCA/WorkloadsOutputReduced/monitor_${jobName}.txt \
experiments/HPCA/WorkloadsOutputReduced/out_${jobName}.txt \
5 \
1800
EOF

            chmod +x "$sbatchFile"

        done
    done
done