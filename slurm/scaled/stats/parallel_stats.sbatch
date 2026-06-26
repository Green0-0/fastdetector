#!/bin/bash
#SBATCH --job-name=parallel_stats
#SBATCH --output=parallel_stats_%A_%a.out
#SBATCH --error=parallel_stats_%A_%a.err
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

export PYTHONUNBUFFERED=1

I=$SLURM_ARRAY_TASK_ID
SHARD_BASE="G-reen_cc-2021-rewritten-split-$I"
SN="${SHARD_BASE}-sn"
TOK="${SHARD_BASE}-tok"
EMB="${SHARD_BASE}-emb"
STAT="${SHARD_BASE}-stat"

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

echo "Processing split $I"

echo "Computing soft ngram scores..."
python scripts/computational/run_softngram_stats.py \
    --source-dataset "$SHARD_BASE" \
    --target-dataset "$SN" \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Computing token embeddings..."
python scripts/computational/run_embedding_tokenwise_stat.py \
    --source-dataset "$SN" \
    --target-dataset "$TOK" \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Computing embeddings..."
python scripts/computational/run_embedding_stats.py \
    --source-dataset "$TOK" \
    --target-dataset "$EMB" \
    --model-name Qwen/Qwen3-Embedding-4B \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Calculating final embedding metrics and removing raw embedding columns..."
python scripts/auxilary/metrics/calc_embedding_metrics.py \
    --source-dataset "$EMB" \
    --columns-prefix original final_response \
    --target-dataset "$STAT" \
    --pairwise-cosim --bertscore --moverscore \
    --remove-columns-afterwards \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Calculating final basic pair metrics..."
python scripts/auxilary/metrics/calc_basic_pair_metrics.py \
    --source-dataset "$STAT" \
    --columns original final_response \
    --target-dataset "$STAT" \
    --jaccards-1 --jaccards-2 --levenshteins \
    --save-locally-instead \
    --cache-dir $CACHE_DIR

echo "Array task $I complete."
