#!/bin/bash
#SBATCH --job-name=calc_stats_array_resume
#SBATCH --output=calc_stats_array_resume_%A_%a.out
#SBATCH --error=calc_stats_array_resume_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1
#SBATCH --array=0-24

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source "$HOME/.cargo/env"

export VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
if [[ ! -d "${VLLM_VENV_PATH}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

I=$SLURM_ARRAY_TASK_ID
BASE="G-reen/cc-2021-rewritten-split-$I"
STAT="${BASE}-stat"

echo "Processing split $I (Resuming)"

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"
# Replace - with _ in safe_name for the dataset cache directory
SAFE_NAME="${STAT//\//_}"
DATASET_PATH="${CACHE_DIR}/${SAFE_NAME}"

# Function to check if a column exists in the dataset
check_column() {
    local col="$1"
    python -c "
import sys
from datasets import load_from_disk
try:
    ds = load_from_disk('$DATASET_PATH')
    if '$col' in ds.column_names:
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    sys.exit(1)
"
}

if ! check_column "pairwise_cosdist_original_final_response"; then
    echo "Resuming: Computing embedding metrics..."
    python scripts/auxilary/metrics/calc_embedding_metrics.py \
        --source-dataset "$STAT" \
        --columns-prefix original final_response \
        --target-dataset "$STAT" \
        --pairwise-cosim --bertscore --moverscore \
        --save-locally-instead \
        --cache-dir "$CACHE_DIR"
else
    echo "Skipping embedding metrics (already completed)."
fi

if ! check_column "original_tokens_llama_perplexity"; then
    echo "Resuming: Computing LLM metrics (original)..."
    python scripts/auxilary/metrics/calc_llm_metrics.py \
        --source-dataset "$STAT" \
        --target-dataset "$STAT" \
        --token-columns original_tokens_llama original_tokens_llama_base \
        --logprob-columns original_top_logprobs_llama original_top_logprobs_llama_base \
        --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
        --save-locally-instead \
        --cache-dir "$CACHE_DIR"
else
    echo "Skipping LLM metrics for 'original' (already completed)."
fi

if ! check_column "final_response_tokens_llama_perplexity"; then
    echo "Resuming: Computing LLM metrics (final_response)..."
    python scripts/auxilary/metrics/calc_llm_metrics.py \
        --source-dataset "$STAT" \
        --target-dataset "$STAT" \
        --token-columns final_response_tokens_llama final_response_tokens_llama_base \
        --logprob-columns final_response_top_logprobs_llama final_response_top_logprobs_llama_base \
        --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
        --save-locally-instead \
        --cache-dir "$CACHE_DIR"
else
    echo "Skipping LLM metrics for 'final_response' (already completed)."
fi

echo "Array task $I complete."
