#!/bin/bash

# Directory containing the sbatch files
SBATCH_DIR="experiments/HPCA/WorkloadsSlurmReduced"

# Check that the directory exists
if [ ! -d "$SBATCH_DIR" ]; then
    echo "Error: directory '$SBATCH_DIR' does not exist."
    exit 1
fi

# Count submitted jobs
count=0

for file in "$SBATCH_DIR"/*.sbatch; do
    # Skip if no files exist
    [ -e "$file" ] || continue

    echo "Submitting $file"
    sbatch "$file"

    ((count++))

    # Small delay to avoid flooding Slurm
    sleep 0.2
done

echo ""
echo "Submitted $count jobs."