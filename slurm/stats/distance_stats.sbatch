#!/bin/bash
#SBATCH --partition=compsci-gpu
#SBATCH --job-name=fd_distance_stats
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-9

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"
DATASET_KIND="${DATASET_KIND:-train}"
case "${DATASET_KIND}" in
  train|val|test) ;;
  *) echo "DATASET_KIND must be train, val, or test." >&2; exit 1 ;;
esac

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
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT

echo "Computing distance stats for ${DATASET_KIND} shard ${SLURM_ARRAY_TASK_ID:-0}..."
python scripts/stats/distance_stats.py \
    --globals-config config/globals.toml \
    --distance-config config/distance_stats.toml \
    --dataset-kind "${DATASET_KIND}" \
    --batch-id "${SLURM_ARRAY_TASK_ID:-0}"

echo "Distance stats complete for shard ${SLURM_ARRAY_TASK_ID:-0}."
