#!/bin/bash
#SBATCH --job-name=calc_stats_split
#SBATCH --output=calc_stats_split_%j.out
#SBATCH --error=calc_stats_split_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

python -c "
import os
from datasets import load_dataset
ds_name = 'G-reen/cc-2021-rewritten'
print(f'Loading {ds_name}...')
ds = load_dataset(ds_name, split='train')
os.makedirs('cached_ds', exist_ok=True)
num_shards = 25
for i in range(num_shards):
    shard = ds.shard(num_shards=num_shards, index=i)
    safe_name = f'G-reen_cc-2021-rewritten-split-{i}'
    local_path = os.path.join('cached_ds', safe_name)
    print(f'Saving shard {i} to {local_path}...')
    shard.save_to_disk(local_path)
print('Splitting done.')
"
