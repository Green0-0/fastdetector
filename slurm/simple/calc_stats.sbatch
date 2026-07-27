#!/bin/bash
#SBATCH --job-name=calc_stats
#SBATCH --output=calc_stats_%j.out
#SBATCH --error=calc_stats_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1

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

# ---- Compute statistics on the generated (stat) dataset ----
echo "Computing distance statistics..."
python scripts/stats/distance_stats.py \
    --globals-config config/globals.toml \
    --distance-config config/distance_stats.toml \
    --batch-id 0

echo "Computing LLM statistics..."
python scripts/stats/llm_stats.py \
    --globals-config config/globals.toml \
    --llm-config config/llm_stats.toml \
    --batch-id 0

echo "Building the evaluation README..."
python scripts/analysis.py \
    --globals-config config/globals.toml \
    --analysis-config config/analysis.toml
echo "Statistics complete."
