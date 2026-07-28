#!/bin/bash
#SBATCH --job-name=fd_gen
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4
#SBATCH --array=0-7

# <prefix>-filtered -> <prefix>-rewritten (+ a clone into <prefix>-stat).
# Task i uses config/gen/shard_i.toml, so the array covers shard_0..shard_7
# only; config/gen/shard_oai.toml is run by hand.

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
if (( BATCH_ID < 0 || BATCH_ID > 7 )); then
  echo "No per-shard gen config for batch ${BATCH_ID} (expected 0-7)." >&2
  exit 1
fi

echo "Generating shard ${BATCH_ID}..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "config/gen/shard_${BATCH_ID}.toml" \
    --batch-id "${BATCH_ID}"

echo "Generation complete for shard ${BATCH_ID}."
