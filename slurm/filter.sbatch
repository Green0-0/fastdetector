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

# ---- Compile caches on node-local disk ----
# Triton defaults to $HOME/.triton. $HOME is NFS here, and concurrent engines
# compiling the same kernels into one shared directory race each other; NFS
# answers with ESTALE ("Stale file handle"), which surfaces as a Triton
# compilation failure and kills the worker. $HOME is also quota-limited, so
# filling it with compile artifacts is undesirable regardless.
# SLURM_JOB_ID is distinct per array task, so each task gets its own root.
FD_CACHE_ROOT="/tmp/fd_${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${FD_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${FD_CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${FD_CACHE_ROOT}/vllm"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT

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
