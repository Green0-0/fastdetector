#!/bin/bash
#SBATCH --job-name=gen_oai_shard
#SBATCH --output=gen_oai_shard_%j.out
#SBATCH --error=gen_oai_shard_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

# Shard 8 and 9 are left untouched from the previous merging (array 0-7)
SHARD_ID=8  
API_URL="" 
MODEL_NAME="gpt-5.4-mini"

SHARD_BASE="G-reen_cc-2021-filtered-split-${SHARD_ID}"
TARGET_DS="G-reen/cc-2021-rewritten-${MODEL_NAME}"

echo "Starting dataset generation for shard ${SHARD_ID} using model ${MODEL_NAME}..."

python scripts/computational/gen_pipeline_oai.py \
    --api-url "$API_URL" \
    --model-name "$MODEL_NAME" \
    --disable-thinking \
    --max-dataset-words 10000 \
    --prompt-file prompts/combined_dataset_test.json \
    --source-dataset "$SHARD_BASE" \
    --source-column collected_subset_original_final_response \
    --num-samples 500 \
    --target-dataset "$TARGET_DS" \
    --cache-dir $CACHE_DIR

echo "Dataset generation complete for shard ${SHARD_ID}."
