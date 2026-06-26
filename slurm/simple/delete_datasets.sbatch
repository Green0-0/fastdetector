#!/bin/bash
#SBATCH --job-name=delete_datasets
#SBATCH --output=delete_datasets_%j.out
#SBATCH --error=delete_datasets_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# ---- Activate project venv ----
source .venv/bin/activate

DATASETS=(
    "G-reen/cc-2021-rewritten-sn"
    "G-reen/cc-2021-rewritten-tok"
    "G-reen/cc-2021-rewritten-emb"
    "G-reen/cc-2021-rewritten-rerank"
    "G-reen/cc-2021-rewritten-llm"
    "G-reen/cc-2021-rewritten-llm2"
    "G-reen/cc-2021-rewritten-stat"
    "G-reen/cc-2021-rewritten-stat-editlens"
)

echo "Deleting datasets..."
for dataset in "${DATASETS[@]}"; do
    echo "Attempting to delete $dataset..."
    hf repos delete "$dataset" --type dataset --yes --missing-ok || echo "Failed to delete $dataset."
done
echo "Cleanup complete!"
