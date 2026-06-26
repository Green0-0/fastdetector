#!/bin/bash
#SBATCH --job-name=parallel_gen
#SBATCH --output=parallel_gen_%A_%a.out
#SBATCH --error=parallel_gen_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4
#SBATCH --array=0-4

source .venv/bin/activate
export PYTHONUNBUFFERED=1

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
I=$SLURM_ARRAY_TASK_ID

case $I in
    0)
        MODEL_NAME="cyankiwi/granite-4.1-8b-AWQ-INT4"
        TEMP=0.6
        TOP_P=0.9
        TOP_K=40
        DISABLE_THINKING=""
        ;;
    1)
        MODEL_NAME="casperhansen/llama-3-8b-instruct-awq"
        TEMP=0.6
        TOP_P=0.9
        TOP_K=40
        DISABLE_THINKING=""
        ;;
    2)
        MODEL_NAME="Qwen/Qwen3-8B-AWQ"
        TEMP=0.7
        TOP_P=0.8
        TOP_K=20
        DISABLE_THINKING="--disable-thinking"
        ;;
    3)
        MODEL_NAME="google/gemma-4-E4B-it"
        TEMP=0.7
        TOP_P=0.8
        TOP_K=-1
        DISABLE_THINKING="--disable-thinking"
        ;;
    4)
        MODEL_NAME="TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        TEMP=0.6
        TOP_P=0.9
        TOP_K=40
        DISABLE_THINKING=""
        ;;
esac

SHARD_BASE="G-reen_cc-2021-filtered-split-$I"
TARGET_DS="${SHARD_BASE}-gen"

echo "Starting dataset generation for shard $I using model $MODEL_NAME..."
python scripts/computational/gen_pipeline.py \
    --model-name "$MODEL_NAME" \
    --temperature $TEMP \
    --top-p $TOP_P \
    --top-k $TOP_K \
    --presence-penalty 1.5 \
    $DISABLE_THINKING \
    --max-model-len 32000 \
    --max-dataset-len 15000 \
    --prompt-file prompts/combined_dataset_test.json \
    --source-dataset "$SHARD_BASE" \
    --source-column collected_subset_original_final_response \
    --num-samples 50000 \
    --target-dataset "$TARGET_DS" \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Dataset generation complete for shard $I."
