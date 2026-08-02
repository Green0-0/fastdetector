#!/bin/bash
#SBATCH --job-name=fd_sweep_editlens
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:2
#SBATCH --array=0-9

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
FD_CACHE_ROOT="/tmp/fd_${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${FD_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${FD_CACHE_ROOT}/inductor"
export HF_DATASETS_CACHE="${FD_CACHE_ROOT}/datasets"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${HF_DATASETS_CACHE}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT

# ---- wandb ----
export WANDB_PROJECT="editlens_model_config_sweep"
export WANDB_DIR="${WANDB_DIR:-${FD_CACHE_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"
if [[ -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
  echo "wandb has no credentials: set WANDB_API_KEY or run 'wandb login'." >&2
  exit 1
fi

TRIALS="${TRIALS:-10}"
STUDY_NAME="${STUDY_NAME:-editlens_model_config_sweep}"
JOURNAL_PATH="${JOURNAL_PATH:-sweeps/${STUDY_NAME}.journal}"

echo "Task ${SLURM_ARRAY_TASK_ID:-0}: ${TRIALS} trials on 2 A5000s -> ${STUDY_NAME}"
python scripts/training/ablate_editlens.py \
    --wandb-project "${WANDB_PROJECT}" \
    --trials "${TRIALS}" \
    --gpus 2 \
    --main-process-port 0 \
    --study-name "${STUDY_NAME}" \
    --journal-path "${JOURNAL_PATH}"

echo "Task ${SLURM_ARRAY_TASK_ID:-0} complete."
