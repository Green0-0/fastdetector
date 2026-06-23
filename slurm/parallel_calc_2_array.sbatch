#!/bin/bash
#SBATCH --job-name=calc_stats_array
#SBATCH --output=calc_stats_array_%A_%a.out
#SBATCH --error=calc_stats_array_%A_%a.err
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
SN="${BASE}-sn"
TOK="${BASE}-tok"
EMB="${BASE}-emb"
RERANK="${BASE}-rerank"
LLM="${BASE}-llm"
LLM2="${BASE}-llm2"
STAT="${BASE}-stat"

echo "Processing split $I"

echo "Computing soft ngram scores..."
python scripts/computational/run_softngram_stats.py \
    --source-dataset "$BASE" \
    --target-dataset "$SN" \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Computing token embeddings..."
python scripts/computational/run_embedding_tokenwise_stat.py \
    --source-dataset "$SN" \
    --target-dataset "$TOK" \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Computing embeddings..."
python scripts/computational/run_embedding_stats.py \
    --source-dataset "$TOK" \
    --target-dataset "$EMB" \
    --model-name Qwen/Qwen3-Embedding-4B \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Computing reranker scores..."
python scripts/computational/run_reranker_stats.py \
    --source-dataset "$EMB" \
    --target-dataset "$RERANK" \
    --model-name Qwen/Qwen3-Reranker-4B \
    --columns original final_response \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Computing LLM logprobs (Llama)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset "$RERANK" \
    --target-dataset "$LLM" \
    --model-name unsloth/Llama-3.2-3B-Instruct \
    --columns original final_response \
    --col-suffix _llama \
    --top-logprobs-k 100 \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Computing LLM logprobs (Llama Base)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset "$LLM" \
    --target-dataset "$LLM2" \
    --model-name unsloth/Llama-3.2-3B \
    --columns original final_response \
    --col-suffix _llama_base \
    --top-logprobs-k 100 \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Calculating final metrics..."
python scripts/auxilary/metrics/calc_basic_pair_metrics.py \
    --source-dataset "$LLM2" \
    --columns original final_response \
    --target-dataset "$STAT" \
    --jaccards-1 --jaccards-2 --levenshteins \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

python scripts/auxilary/metrics/calc_embedding_metrics.py \
    --source-dataset "$STAT" \
    --columns-prefix original final_response \
    --target-dataset "$STAT" \
    --pairwise-cosim --bertscore --moverscore \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset "$STAT" \
    --target-dataset "$STAT" \
    --token-columns original_tokens_llama original_tokens_llama_base \
    --logprob-columns original_top_logprobs_llama original_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset "$STAT" \
    --target-dataset "$STAT" \
    --token-columns final_response_tokens_llama final_response_tokens_llama_base \
    --logprob-columns final_response_top_logprobs_llama final_response_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
    --save-locally-instead \
    --cache-dir /usr/project/xtmp/ccprocessing_cache/

echo "Array task $I complete."
