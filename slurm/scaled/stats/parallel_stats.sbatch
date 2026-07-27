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
#SBATCH --array=0-7

# ---- Activate project venv ----
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source "$HOME/.cargo/env"

# ---- vLLM venv (launched as a subprocess by the pipeline) ----
export VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
if [[ ! -d "${VLLM_VENV_PATH}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

# ---- HF cache (replaces the old scripts' --cache-dir flag) ----
export HF_HOME="${HF_HOME:-/usr/project/xtmp/ccprocessing_cache/}"

# ---- One statistics job per shard of the stat dataset ----
# The array range must match the shard configs uploaded by the gen stage:
# 0-7 covers the parallel_gen array. If the OAI shards were also generated
# (gen_oai_shard.sbatch uploads shard 8, and 9 if run twice), extend the
# range above accordingly, e.g. --array=0-8.
# Stats scripts are idempotent per column: reruns skip already-computed
# metrics, so a partially failed array can simply be resubmitted.
I=$SLURM_ARRAY_TASK_ID
echo "Processing shard $I"

echo "Computing distance statistics..."
python scripts/stats/distance_stats.py \
    --globals-config config/globals.toml \
    --distance-config config/distance_stats.toml \
    --batch-id "$I"

echo "Computing LLM statistics..."
python scripts/stats/llm_stats.py \
    --globals-config config/globals.toml \
    --llm-config config/llm_stats.toml \
    --batch-id "$I"

echo "Shard $I complete."
