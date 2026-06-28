#!/bin/bash
#SBATCH --job-name=merge_ds
#SBATCH --output=merge_ds_%j.out
#SBATCH --error=merge_ds_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source .venv/bin/activate
export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"
DATASET_NAME="G-reen/cc-2021-rewritten-stat"

echo "Merging split datasets..."
python scripts/auxilary/dataset/merge_datasets.py \
    --source_dataset G-reen/cc-2021-rewritten \
    --target_dataset $DATASET_NAME \
    --num_shards 25 \
    --shard_suffix="-stat" \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Calculating norms (minimax and quantiles)..."
python scripts/auxilary/metrics/calc_norms.py \
    --source-dataset $DATASET_NAME \
    --target-dataset $DATASET_NAME \
    --minimax pairwise_cosdist_original_final_response levenshtein_original_final_response jaccard_1_original_final_response jaccard_2_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --quantile pairwise_cosdist_original_final_response levenshtein_original_final_response jaccard_1_original_final_response jaccard_2_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Building readme on merged dataset..."
python scripts/auxilary/metrics/build_readme.py \
    --source-dataset $DATASET_NAME \
    --target-dataset $DATASET_NAME \
    --text-columns-analyze original final_response \
    --summary-stat-columns pairwise_cosdist_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --pairwise-correlations levenshtein_original_final_response_quantile jaccard_1_original_final_response_quantile pairwise_cosdist_original_final_response_quantile pairwise_softngram_original_final_response_quantile pairwise_bertscore_f1_original_final_response_quantile pairwise_moverscore_original_final_response_quantile \
    --histogram-columns pairwise_cosdist_original_final_response levenshtein_original_final_response jaccard_1_original_final_response/jaccard_2_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --scatterplot-columns pairwise_cosdist_original_final_response/pairwise_softngram_original_final_response \
    --save-locally-instead \
    --cache-dir $CACHE_DIR
    
echo "All complete!"
