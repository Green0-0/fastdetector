#!/bin/bash
#SBATCH --job-name=split_ds
#SBATCH --output=split_ds_%j.out
#SBATCH --error=split_ds_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

python scripts/auxilary/dataset/split_dataset.py \
    --source_dataset G-reen/cc-2021-rewritten \
    --num_shards 25 \
    --cache-dir $CACHE_DIR
