#!/bin/bash
#SBATCH --job-name=editlens_stats
#SBATCH --output=editlens_stats_%j.out
#SBATCH --error=editlens_stats_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1

# ---- Activate project venv ----
source .venv/bin/activate

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source "$HOME/.cargo/env"

# ---- vLLM venv ----
export VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"

if [[ ! -d "${VLLM_VENV_PATH}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

# ---- Run the dataset analysis ----
python scripts/computational/run_editlens_stat.py --source-dataset G-reen/cc-2021-rewritten --fastdetector-prompt-metadata-column prompt
