#!/bin/bash
# Submit the full CANE main table (5 datasets x 2 backbones x 5 seeds = 50 jobs)
# to SLURM general-gpu. Each job runs one config from cane/configs/.
#   bash cane/run_all.sh            # submit all 50
#   bash cane/run_all.sh cora       # submit only cora cells
CANE=/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/cane
filter="${1:-}"
n=0
for cfg in "$CANE"/configs/wbXmod_*.json; do
  name=$(basename "$cfg" .json)
  if [ -n "$filter" ] && [[ "$name" != *"_${filter}_"* ]]; then continue; fi
  sbatch --job-name="cane_${name#wbXmod_}" --export=ALL,CFG="$name" "$CANE/run.sbatch"
  n=$((n+1))
done
echo "submitted $n job(s)"
