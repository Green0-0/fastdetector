#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --job-name=fd_gen
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:rtx_pro_6000:2
#SBATCH --array=0-8

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

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

GEN_CONFIG="config/gen/shard_${SLURM_ARRAY_TASK_ID:-0}.toml"

if [[ ! -f "${GEN_CONFIG}" ]]; then
  echo "No gen config at ${GEN_CONFIG}." >&2
  exit 1
fi

# ---- Engine venv (launched as a subprocess by the pipeline) ----
ENGINE=$(sed -n 's/^[[:space:]]*engine[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${GEN_CONFIG}" | head -1)
case "${ENGINE}" in
  vllm|aphrodite)
    if [[ ! -x ".${ENGINE}/bin/${ENGINE}" ]]; then
      echo "${ENGINE} venv not found at .${ENGINE}; see README.md" >&2
      exit 1
    fi
    ;;
  "")
    echo "Could not read an engine from ${GEN_CONFIG}." >&2
    exit 1
    ;;
esac

export FD_ENGINE_HF_OFFLINE=1
sleep $(( ${SLURM_ARRAY_TASK_ID:-0} * 60 ))

echo "Generating shard ${SLURM_ARRAY_TASK_ID:-0}..."
fd_run_requeueable python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "config/gen/shard_${SLURM_ARRAY_TASK_ID:-0}.toml" \
    --batch-id "${SLURM_ARRAY_TASK_ID:-0}"

echo "Generation complete for shard ${SLURM_ARRAY_TASK_ID:-0}."
