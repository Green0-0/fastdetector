#!/bin/bash
#SBATCH --job-name=fd_editlens_stats
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-7

# EditLens buckets/scores, added in place to <prefix>-stat[shard_i].
# The three stats stages rewrite the same shard, so never run two at once.

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

BATCH_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Computing EditLens stats for shard ${BATCH_ID}..."
python scripts/stats/editlens_stats.py \
    --globals-config config/globals.toml \
    --editlens-config config/editlens_stats.toml \
    --batch-id "${BATCH_ID}"

echo "EditLens stats complete for shard ${BATCH_ID}."
