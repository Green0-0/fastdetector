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

# ---- Run the dataset analysis ----
echo "Computing soft ngram scores..."
python scripts/computational/run_softngram_stats.py \
    --source-dataset G-reen/cc-2021-rewritten \
    --target-dataset G-reen/cc-2021-rewritten-sn \
    --columns original final_response

echo "Computing token embeddings..."
python scripts/computational/run_embedding_tokenwise_stat.py \
    --source-dataset G-reen/cc-2021-rewritten-sn \
    --target-dataset G-reen/cc-2021-rewritten-tok \
    --columns original final_response
    
echo "Computing embeddings..."
python scripts/computational/run_embedding_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-tok \
    --target-dataset G-reen/cc-2021-rewritten-emb \
    --model-name Qwen/Qwen3-Embedding-4B \
    --columns original final_response

echo "Computing reranker scores..."
python scripts/computational/run_reranker_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-emb \
    --target-dataset G-reen/cc-2021-rewritten-rerank \
    --model-name Qwen/Qwen3-Reranker-4B \
    --columns original final_response

echo "Computing LLM logprobs (Llama)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-rerank \
    --target-dataset G-reen/cc-2021-rewritten-llm \
    --model-name unsloth/Llama-3.2-3B-Instruct \
    --columns original final_response \
    --col-suffix _llama \
    --top-logprobs-k 100

echo "Computing LLM logprobs (Llama Base)..."
python scripts/computational/run_llm_stats.py \
    --source-dataset G-reen/cc-2021-rewritten-llm \
    --target-dataset G-reen/cc-2021-rewritten-llm2 \
    --model-name unsloth/Llama-3.2-3B \
    --columns original final_response \
    --col-suffix _llama_base \
    --top-logprobs-k 100

echo "Calculating final metrics..."
python scripts/auxilary/metrics/calc_basic_pair_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-llm2 \
    --columns original final_response \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --jaccards-1 --jaccards-2 --levenshteins

python scripts/auxilary/metrics/calc_embedding_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --columns original final_response \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --pairwise-cosim --bertscore --moverscore

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --column original \
    --token-columns original_tokens_llama original_tokens_llama_base \
    --logprob-columns original_top_logprobs_llama original_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score

python scripts/auxilary/metrics/calc_llm_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --column final_response \
    --token-columns final_response_tokens_llama final_response_tokens_llama_base \
    --logprob-columns final_response_top_logprobs_llama final_response_top_logprobs_llama_base \
    --perplexity --entropy --topp-outlier --topk-outlier --binoculars-score --fastdetectgpt-score

python scripts/auxilary/metrics/calc_quantiles.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --columns pairwise_cossim_original_final_response levenshtein_original_final_response jaccard_1_original_final_response jaccard_2_original_final_response pairwise_cross_encoder_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response

python scripts/auxilary/metrics/calc_minimax_norms.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --columns levenshtein_original_final_response pairwise_cross_encoder_original_final_response

python scripts/auxilary/metrics/build_readme.py \
    --source-dataset G-reen/cc-2021-rewritten-stat \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --text-columns-analyze original final_response \
    --summary-stat-columns original_tokens_llama_perplexity final_response_tokens_llama_perplexity original_tokens_llama_base_perplexity final_response_tokens_llama_base_perplexity original_tokens_llama_entropy final_response_tokens_llama_entropy original_tokens_llama_base_entropy final_response_tokens_llama_base_entropy original_tokens_llama_fastdetectgpt final_response_tokens_llama_fastdetectgpt original_tokens_llama_base_fastdetectgpt final_response_tokens_llama_base_fastdetectgpt original_tokens_llama_topp_outlier final_response_tokens_llama_topp_outlier original_tokens_llama_base_topp_outlier final_response_tokens_llama_base_topp_outlier original_tokens_llama_topk_outlier final_response_tokens_llama_topk_outlier original_tokens_llama_base_topk_outlier final_response_tokens_llama_base_topk_outlier original_tokens_llama_binoculars final_response_tokens_llama_binoculars pairwise_cossim_original_final_response pairwise_cross_encoder_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --pairwise-average-differences levenshtein_original_final_response_quantile pairwise_cross_encoder_original_final_response_quantile jaccard_1_original_final_response_quantile pairwise_cossim_original_final_response_quantile pairwise_softngram_original_final_response_quantile pairwise_bertscore_f1_original_final_response_quantile pairwise_moverscore_original_final_response_quantile \
    --histogram-columns original_tokens_llama_perplexity/final_response_tokens_llama_perplexity/original_tokens_llama_base_perplexity/final_response_tokens_llama_base_perplexity original_tokens_llama_entropy/final_response_tokens_llama_entropy/original_tokens_llama_base_entropy/final_response_tokens_llama_base_entropy original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt/original_tokens_llama_base_fastdetectgpt/final_response_tokens_llama_base_fastdetectgpt original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier/original_tokens_llama_base_topp_outlier/final_response_tokens_llama_base_topp_outlier original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier/original_tokens_llama_base_topk_outlier/final_response_tokens_llama_base_topk_outlier original_tokens_llama_binoculars/final_response_tokens_llama_binoculars pairwise_cossim_original_final_response pairwise_cross_encoder_original_final_response levenshtein_original_final_response jaccard_1_original_final_response/jaccard_2_original_final_response pairwise_softngram_original_final_response pairwise_bertscore_f1_original_final_response pairwise_moverscore_original_final_response \
    --scatterplot-columns pairwise_softngram_original_final_response/original_tokens_llama_binoculars/final_response_tokens_llama_binoculars pairwise_softngram_original_final_response/original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt pairwise_cossim_original_final_response/original_tokens_llama_binoculars/final_response_tokens_llama_binoculars pairwise_cossim_original_final_response/original_tokens_llama_fastdetectgpt/final_response_tokens_llama_fastdetectgpt pairwise_softngram_original_final_response/original_tokens_llama_perplexity/final_response_tokens_llama_perplexity pairwise_softngram_original_final_response/original_tokens_llama_entropy/final_response_tokens_llama_entropy pairwise_softngram_original_final_response/original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier pairwise_softngram_original_final_response/original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier pairwise_cossim_original_final_response/original_tokens_llama_perplexity/final_response_tokens_llama_perplexity pairwise_cossim_original_final_response/original_tokens_llama_entropy/final_response_tokens_llama_entropy pairwise_cossim_original_final_response/original_tokens_llama_topp_outlier/final_response_tokens_llama_topp_outlier pairwise_cossim_original_final_response/original_tokens_llama_topk_outlier/final_response_tokens_llama_topk_outlier pairwise_cossim_original_final_response/pairwise_softngram_original_final_response \
    --classifier-columns original_tokens_llama_fastdetectgpt:false/final_response_tokens_llama_fastdetectgpt:true original_tokens_llama_base_fastdetectgpt:false/final_response_tokens_llama_base_fastdetectgpt:true original_tokens_llama_binoculars:true/final_response_tokens_llama_binoculars:false
