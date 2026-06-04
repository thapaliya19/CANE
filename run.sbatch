#!/bin/bash
# Run one CANE experiment on SLURM general-gpu.
# Usage:  sbatch --export=ALL,CFG=wbXmod_cora_gcn_s0 cane/run.sbatch
#   (CFG is the config basename without .json, under cane/configs/)
#SBATCH --job-name=cane
#SBATCH --partition=general-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/cane/logs/%x_%j.out
#SBATCH --error=/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/cane/logs/%x_%j.err

set -e
CANE=/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/cane
source /gpfs/scratchfs1/jhf24001/ldn24004/new_noise/autoresearch/activate_env.sh 2>/dev/null || true
# Make the ported tree the only import root (the runner also inserts $CANE at
# sys.path[0], so cane/src wins over any repo-wide src on PYTHONPATH).
export PYTHONPATH="$CANE:$PYTHONPATH"
cd "$CANE"
mkdir -p logs autoresearch/results
: "${CFG:?set CFG to a config basename, e.g. CFG=wbXmod_cora_gcn_s0}"
echo "[cane] running configs/${CFG}.json"
python3 -u autoresearch/experiment_runner.py --config "configs/${CFG}.json"
