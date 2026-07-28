#!/bin/bash
#SBATCH --job-name=fd_filter
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4
#SBATCH --array=0-7

# <prefix>-raw -> <prefix>-processed -> <prefix>-filtered, one shard per task.

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

# ---- vLLM venv (launched as a subprocess by the pipeline) ----
if [[ ! -x .vllm/bin/vllm ]]; then
  echo "vLLM venv not found at .vllm; see README.md" >&2
  exit 1
fi

BATCH_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Filtering shard ${BATCH_ID}..."
python scripts/filter.py \
    --globals-config config/globals.toml \
    --filter-config config/filter.toml \
    --batch-id "${BATCH_ID}"

echo "Filtering complete for shard ${BATCH_ID}."
