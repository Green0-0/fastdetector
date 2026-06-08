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
echo "Computing embeddings..."
python scripts/computational/run_embedding_stats.py \
    --source-dataset G-reen/cc-2021-rewritten \
    --target-dataset G-reen/cc-2021-rewritten-emb \
    --model-name Qwen/Qwen3-Embedding-4B \
    --columns original,final_response

echo "Computing reranker scores..."
python scripts/computational/run_reranker_stat.py \
    --source-dataset G-reen/cc-2021-rewritten-emb \
    --target-dataset G-reen/cc-2021-rewritten-rerank \
    --model-name Qwen/Qwen3-Reranker-4B \
    --col-a original \
    --col-b final_response

echo "Computing LLM logprobs (Llama)..."
python scripts/computational/run_llm_stat.py \
    --source-dataset G-reen/cc-2021-rewritten-rerank \
    --target-dataset G-reen/cc-2021-rewritten-llm \
    --model-name unsloth/Llama-3.2-3B-Instruct \
    --columns original,final_response \
    --col-suffix _llama \
    --top-logprobs-k 100

echo "Computing LLM logprobs (Gemma)..."
python scripts/computational/run_llm_stat.py \
    --source-dataset G-reen/cc-2021-rewritten-llm \
    --target-dataset G-reen/cc-2021-rewritten-llm2 \
    --model-name google/gemma-3-4b-it \
    --columns original,final_response \
    --col-suffix _gemma \
    --top-logprobs-k 100

echo "Calculating final metrics..."
python scripts/auxilary/calculate_metrics.py \
    --source-dataset G-reen/cc-2021-rewritten-llm2 \
    --target-dataset G-reen/cc-2021-rewritten-stat \
    --col-human original \
    --col-ai final_response
