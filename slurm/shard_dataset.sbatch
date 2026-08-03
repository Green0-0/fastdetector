#!/bin/bash
#SBATCH --job-name=fd_shard_dataset
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

echo "Sharding dataset..."
python scripts/shard_dataset.py \
    --globals-config config/globals.toml \
    --source-dataset "G-reen/cc-2021-raw"

echo "Sharding complete."




