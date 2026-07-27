#!/bin/bash
#SBATCH --job-name=parallel_gen
#SBATCH --output=parallel_gen_%A_%a.out
#SBATCH --error=parallel_gen_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4
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

# Aphrodite venv is needed only by the aphrodite-engine shard configs.
export APHRODITE_VENV_PATH="${APHRODITE_VENV_PATH:-${PWD}/.aphrodite}"

# ---- One generation job per shard ----
# Requires the filter stage to have uploaded shard configs: set
# output_shards in config/filter.toml to at least the array size (the
# original layout used 10 shards, reserving 8/9 for OAI generation).
# Each config/gen/shard_<I>.toml pins the model, engine, and sampling
# params for that shard; --batch-id <I> selects the shard_<I> config of
# the filtered dataset.
I=$SLURM_ARRAY_TASK_ID
echo "Starting dataset generation for shard $I..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "config/gen/shard_${I}.toml" \
    --batch-id "$I"
echo "Dataset generation complete for shard $I."
