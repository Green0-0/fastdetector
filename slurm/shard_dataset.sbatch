#!/bin/bash
#SBATCH --job-name=fd_shard_dataset
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Splits <prefix>-raw into the shard_<i> configs every later stage reads.
# This is where the size of a run is set; everything downstream processes
# whatever its shard holds.

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-5000000}"

echo "Sharding ${NUM_SAMPLES} rows into ${NUM_SHARDS} shards..."
python scripts/shard_dataset.py \
    --globals-config config/globals.toml \
    --num-shards "${NUM_SHARDS}" \
    --num-samples "${NUM_SAMPLES}"

echo "Sharding complete."
