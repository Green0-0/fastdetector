#!/bin/bash
#SBATCH --job-name=gen_oai_shard
#SBATCH --output=gen_oai_shard_%j.out
#SBATCH --error=gen_oai_shard_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ---- Activate project venv ----
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# ---- HF cache (replaces the old scripts' --cache-dir flag) ----
export HF_HOME="${HF_HOME:-/usr/project/xtmp/ccprocessing_cache/}"

# ---- API key ----
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# ---- OAI generation for a reserved shard ----
# Shards 8 and 9 are reserved for API-based generation when the filter
# stage was run with output_shards = 10 (the parallel_gen array covers
# 0-7). api_url, api_key_env, and the model come from shard_oai.toml.
SHARD_ID=8
echo "Starting OAI dataset generation for shard ${SHARD_ID}..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config config/gen/shard_oai.toml \
    --batch-id "$SHARD_ID"
echo "Dataset generation complete for shard ${SHARD_ID}."
