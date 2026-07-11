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

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

# ---- Run the dataset generation ----
echo "Starting dataset generation..."
python scripts/computational/gen_pipeline.py \
    --model-name google/gemma-4-E4B-it \
    --temperature 0.0 \
    --top-p 1.0 \
    --presence-penalty 0.0 \
    --max-model-len 32000 \
    --max-dataset-len 15000 \
    --prompt-file prompts/filter_contiguous_subset.json \
    --source-dataset G-reen/cc-2021-raw \
    --source-column trafilatura_text \
    --num-samples 400000 \
    --target-dataset G-reen/cc-2021-processed \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_basic_pair_metrics.py \
    --source-dataset G-reen/cc-2021-processed \
    --columns original final_response \
    --target-dataset G-reen/cc-2021-processed-stats \
    --deviated-lines --deviated-words --deviated-characters --loose-subset-collect --strict-subset \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_quantiles.py \
    --source-dataset G-reen/cc-2021-processed-stats \
    --columns deviated_lines_proportion_original_final_response deviated_words_proportion_original_final_response deviated_characters_proportion_original_final_response \
    --target-dataset G-reen/cc-2021-processed-stats \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/build_readme.py \
    --source-dataset G-reen/cc-2021-processed-stats \
    --target-dataset G-reen/cc-2021-processed-stats \
    --summary-stat-columns is_loose_subset_original_final_response is_strict_subset_original_final_response deviated_lines_proportion_original_final_response deviated_words_proportion_original_final_response deviated_characters_proportion_original_final_response deviated_lines_original_final_response deviated_words_original_final_response deviated_characters_original_final_response \
    --histogram-columns deviated_lines_original_final_response deviated_words_original_final_response deviated_characters_original_final_response deviated_lines_proportion_original_final_response deviated_words_proportion_original_final_response deviated_characters_proportion_original_final_response \
    --cache-dir $CACHE_DIR

python scripts/auxilary/dataset/subset_data.py \
    --source_dataset G-reen/cc-2021-processed-stats \
    --target_dataset G-reen/cc-2021-filtered \
    --conditions "is_loose_subset_original_final_response=True,deviated_lines_proportion_original_final_response_quantile<0.9,deviated_words_proportion_original_final_response_quantile<0.9,deviated_characters_proportion_original_final_response_quantile<0.9" \
    --cache-dir $CACHE_DIR

echo "Dataset generation complete."
