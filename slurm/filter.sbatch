#!/bin/bash
#SBATCH --partition=compsci-gpu
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
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.3/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"

# ---- Compile caches on node-local disk ----
FD_CACHE_ROOT="/tmp/fd_${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${FD_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${FD_CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${FD_CACHE_ROOT}/vllm"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT
source slurm/resume_job.sh

# ---- vLLM venv (launched as a subprocess by the pipeline) ----
if [[ ! -x .vllm/bin/vllm ]]; then
  echo "vLLM venv not found at .vllm; see README.md" >&2
  exit 1
fi

export FD_ENGINE_HF_OFFLINE=1
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 60 ))

echo "Filtering shard ${SLURM_ARRAY_TASK_ID:-0}..."
fd_run_requeueable python scripts/filter.py \
    --globals-config config/globals.toml \
    --filter-config config/filter.toml \
    --batch-id "${SLURM_ARRAY_TASK_ID:-0}"

echo "Filtering complete for shard ${SLURM_ARRAY_TASK_ID:-0}."
