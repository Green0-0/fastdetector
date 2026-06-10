#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=gen-data-%j.out
#SBATCH --error=gen-data-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4

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

# ---- Run the dataset generation ----
echo "Starting dataset generation..."
python scripts/computational/gen_pipeline.py \
    --model-name google/gemma-4-E4B-it \
    --temperature 0.7 \
    --top-p 0.8 \
    --presence-penalty 1.5 \
    --max-model-len 32000 \
    --max-dataset-len 15000 \
    --prompt-file prompts/combined_dataset.json \
    --source-dataset G-reen/cc-2021-filtered \
    --source-column collected_subset_original_final_response \
    --num-samples 1000 \
    --target-dataset G-reen/cc-2021-rewritten

echo "Dataset generation complete."
