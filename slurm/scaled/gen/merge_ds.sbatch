#!/bin/bash
#SBATCH --job-name=merge_gen_ds
#SBATCH --output=merge_gen_ds_%j.out
#SBATCH --error=merge_gen_ds_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"
DATASET_NAME="G-reen/cc-2021-rewritten"

echo "Merging 8 generated shards..."
python scripts/auxilary/dataset/merge_datasets.py \
    --source_dataset G-reen/cc-2021-filtered \
    --target_dataset $DATASET_NAME \
    --num_shards 8 \
    --shard_suffix -gen \
    --cache-dir $CACHE_DIR

echo "Dataset merge and upload complete!"
