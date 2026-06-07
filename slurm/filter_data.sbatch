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
python scripts/computational/run_pipeline.py \
    --model-name google/gemma-4-E4B-it \
    --temperature 0.0 \
    --top-p 1.0 \
    --presence-penalty 0.0 \
    --max-model-len 32000 \
    --max-dataset-len 15000 \
    --prompt-file prompts/filter_contiguous_subset.json \
    --source-dataset G-reen/cc-2021-raw \
    --source-column trafilatura_text \
    --num-samples 1000 \
    --target-dataset G-reen/cc-2021-processed

python scripts/auxilary/filter_data_pairs.py \
    --source-dataset G-reen/cc-2021-processed \
    --original-column original \
    --new-column final_response \
    --target-dataset G-reen/cc-2021-processed-stats

python scripts/auxilary/subset_data.py \
    --source_dataset G-reen/cc-2021-processed-stats \
    --target_dataset G-reen/cc-2021-filtered \
    --conditions "is_loose_subset=True,deviated_lines_proportion_quantile<0.9,deviated_words_proportion_quantile<0.9,deviated_characters_proportion_quantile<0.9"

echo "Dataset generation complete."
