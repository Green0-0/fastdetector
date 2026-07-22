#!/bin/bash
#SBATCH --job-name=calc_stats
#SBATCH --output=calc_stats_%j.out
#SBATCH --error=calc_stats_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1

# ---- Activate project venv ----
source .venv/bin/activate

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

export PYTHONUNBUFFERED=1

CACHE_DIR="/usr/project/xtmp/ccprocessing_cache/"

# ---- Run the dataset analysis ----
echo "Computing soft ngram scores..."
python scripts/computational/run_softngram_stats.py \
    --source-dataset G-reen/cc-2021-rewritten \
    --target-dataset G-reen/cc-2021-rewritten-sn \
    --columns original final_response \
    --cache-dir $CACHE_DIR

echo "Computing token embeddings..."
python scripts/computational/run_embedding_tokenwise_stat.py \
    --source-dataset G-reen/cc-2021-rewritten-sn \
    --target-dataset G-reen/cc-2021-rewritten-tok \
    --columns original final_response \
    --cache-dir $CACHE_DIR
    
echo "Computing embeddings..."
python scripts/computational/run_embedding_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-tok \
    --target-dataset G-reen/cc-2021-rewritten-emb \
    --model-name Qwen/Qwen3-Embedding-4B \
    --columns original final_response \
    --cache-dir $CACHE_DIR

echo "Computing LLM logprobs (Llama)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-emb \
    --target-dataset G-reen/cc-2021-rewritten-llm \
    --model-name unsloth/Llama-3.2-3B-Instruct \
    --columns original final_response \
    --col-suffix _llama \
    --top-logprobs-k 100 \
    --cache-dir $CACHE_DIR

echo "Computing LLM logprobs (Llama Base)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-llm \
    --target-dataset G-reen/cc-2021-rewritten-llm2 \
    --model-name unsloth/Llama-3.2-3B \
    --columns original final_response \
    --col-suffix _llama_base \
    --top-logprobs-k 100 \
    --cache-dir $CACHE_DIR

echo "Calculating final metrics..."
python scripts/auxilary/metrics/calc_basic_pair_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-llm2 \
    --columns original final_response \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --jaccard-1 --jaccard-2 --levenshtein \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_embedding_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --columns-prefix original final_response \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --cosdist --bertscore --moverscore \
    --remove-columns-afterwards \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --token-columns original_tokens_llama original_tokens_llama_base \
    --logprob-columns original_top_logprobs_llama original_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
    --remove-columns-afterwards \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --token-columns final_response_tokens_llama final_response_tokens_llama_base \
    --logprob-columns final_response_top_logprobs_llama final_response_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score \
    --remove-columns-afterwards \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/calc_norms.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --minimax cosdist levenshtein jaccard_1 jaccard_2 softngram bertscore moverscore \
    --quantile cosdist levenshtein jaccard_1 jaccard_2 softngram bertscore moverscore \
    --cache-dir $CACHE_DIR

python scripts/auxilary/metrics/build_readme.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --text-columns-analyze original final_response \
    --summary-stat-columns original_tokens_llama_perplexity final_response_tokens_llama_perplexity original_tokens_llama_base_perplexity final_response_tokens_llama_base_perplexity original_tokens_llama_entropy final_response_tokens_llama_entropy original_tokens_llama_base_entropy final_response_tokens_llama_base_entropy original_tokens_llama_fastdetectgpt final_response_tokens_llama_fastdetectgpt original_tokens_llama_base_fastdetectgpt final_response_tokens_llama_base_fastdetectgpt original_tokens_llama_topp_outlier final_response_tokens_llama_topp_outlier original_tokens_llama_base_topp_outlier final_response_tokens_llama_base_topp_outlier original_tokens_llama_topk_outlier final_response_tokens_llama_topk_outlier original_tokens_llama_base_topk_outlier final_response_tokens_llama_base_topk_outlier original_tokens_llama_binoculars final_response_tokens_llama_binoculars cosdist softngram bertscore moverscore \
    --pairwise-correlations levenshtein_quantile jaccard_1_quantile cosdist_quantile softngram_quantile bertscore_quantile moverscore_quantile \
    --histogram-columns original_tokens_llama_perplexity/final_response_tokens_llama_perplexity/original_tokens_llama_base_perplexity/final_response_tokens_llama_base_perplexity original_tokens_llama_entropy/final_response_tokens_llama_entropy/original_tokens_llama_base_entropy/final_response_tokens_llama_base_entropy original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt/original_tokens_llama_base_fastdetectgpt/final_response_tokens_llama_base_fastdetectgpt original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier/original_tokens_llama_base_topp_outlier/final_response_tokens_llama_base_topp_outlier original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier/original_tokens_llama_base_topk_outlier/final_response_tokens_llama_base_topk_outlier original_tokens_llama_binoculars/final_response_tokens_llama_binoculars cosdist levenshtein jaccard_1/jaccard_2 softngram bertscore moverscore \
    --scatterplot-columns softngram/original_tokens_llama_binoculars/final_response_tokens_llama_binoculars softngram/original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt cosdist/original_tokens_llama_binoculars/final_response_tokens_llama_binoculars cosdist/original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt softngram/original_tokens_llama_perplexity/final_response_tokens_llama_perplexity softngram/original_tokens_llama_entropy/final_response_tokens_llama_entropy softngram/original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier softngram/original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier cosdist/original_tokens_llama_perplexity/final_response_tokens_llama_perplexity cosdist/original_tokens_llama_entropy/final_response_tokens_llama_entropy cosdist/original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier cosdist/original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier cosdist/softngram \
    --classifier-columns original_tokens_llama_fastdetectgpt:false/final_response_tokens_llama_fastdetectgpt:true original_tokens_llama_base_fastdetectgpt:false/final_response_tokens_llama_base_fastdetectgpt:true original_tokens_llama_binoculars:true/final_response_tokens_llama_binoculars:false \
    --cache-dir $CACHE_DIR
