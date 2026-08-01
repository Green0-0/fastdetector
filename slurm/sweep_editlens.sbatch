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

# Ten workers on two A5000s each, all driving one optuna study.
#
# The array is the parallelism: every task opens the same journal file, so the
# TPE sampler (constant_liar) hands out different parameters to each and the
# Hyperband pruner rungs them against each other. Trial numbers are allocated
# by the journal, so the wandb run names and the saved adapter directories stay
# distinct without the tasks having to coordinate.
#
#   sbatch slurm/sweep_editlens.sbatch                  # 10 trials per task
#   TRIALS=4 sbatch slurm/sweep_editlens.sbatch         # 40 total instead
#
# Resuming is the same command: an existing study is loaded, not replaced.

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
# See gen.sbatch: $HOME is NFS, and concurrent tasks compiling the same kernels
# into one shared directory race each other into ESTALE. SLURM_JOB_ID is
# distinct per array task, so each task gets its own root.
FD_CACHE_ROOT="/tmp/fd_${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${FD_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${FD_CACHE_ROOT}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT

# ---- wandb ----
# Fail here rather than 40 minutes into the first trial, which is where the
# credentials are first needed.
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
# --main-process-port 0 makes each launch take a free port. Two array tasks can
# land on one node, and accelerate's default 29500 is a hard error for whichever
# of them starts second.
python scripts/training/ablate_editlens.py \
    --wandb-project "${WANDB_PROJECT}" \
    --trials "${TRIALS}" \
    --gpus 2 \
    --main-process-port 0 \
    --study-name "${STUDY_NAME}" \
    --journal-path "${JOURNAL_PATH}"

echo "Task ${SLURM_ARRAY_TASK_ID:-0} complete."
