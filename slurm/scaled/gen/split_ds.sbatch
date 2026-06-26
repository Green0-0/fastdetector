#!/bin/bash
#SBATCH --job-name=split_gen_ds
#SBATCH --output=split_gen_ds_%j.out
#SBATCH --error=split_gen_ds_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

echo "Splitting filtered dataset into 10 shards..."
python scripts/auxilary/dataset/split_dataset.py \
    --source_dataset G-reen/cc-2021-filtered \
    --num_shards 10 \
    --cache-dir $CACHE_DIR

echo "Dataset split complete."
