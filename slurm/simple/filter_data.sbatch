#!/bin/bash
#SBATCH --job-name=filter_data
#SBATCH --output=filter-data-%j.out
#SBATCH --error=filter-data-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4

# ---- Activate project venv ----
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source "$HOME/.cargo/env"

# ---- vLLM venv (launched as a subprocess by the pipeline) ----
export VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
if [[ ! -d "${VLLM_VENV_PATH}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

# ---- HF cache (replaces the old scripts' --cache-dir flag) ----
export HF_HOME="${HF_HOME:-/usr/project/xtmp/ccprocessing_cache/}"

# ---- Run the filtering pipeline ----
# Dataset names, model, sampling params, filter conditions, and output
# sharding all come from config/globals.toml + config/filter.toml.
echo "Starting dataset filtering..."
python scripts/filter.py \
    --globals-config config/globals.toml \
    --filter-config config/filter.toml
echo "Dataset filtering complete."
