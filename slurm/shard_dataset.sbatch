#!/bin/bash
#SBATCH --job-name=fd_shard_dataset
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

# Splits an unsharded corpus into the shard_<i> configs every later stage
# reads. This is where the size of a run is set; everything downstream
# processes whatever its shard holds.
#
#   SOURCE_DATASET=<org>/<corpus> sbatch slurm/shard_dataset.sbatch

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

NUM_SHARDS="${NUM_SHARDS:-8}"

# This stage is driven by arguments rather than a TOML. The source corpus has
# no home in globals.toml, so it has to be passed in; the target is the raw
# dataset globals.toml already names, read back here so the two cannot drift.
SOURCE_DATASET="${SOURCE_DATASET:-}"
if [[ -z "${SOURCE_DATASET}" ]]; then
  echo "Set SOURCE_DATASET to the unsharded corpus to split, e.g." >&2
  echo "  SOURCE_DATASET=<org>/<corpus> sbatch slurm/shard_dataset.sbatch" >&2
  exit 1
fi

TARGET_DATASET="${TARGET_DATASET:-$(python -c '
from fastdetector.frontend.toml_config import GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml

globals_config = GlobalsConfig(**load_toml("config/globals.toml"))
print(globals_config.resolve_dataset(globals_config.raw_dataset))
')}"

echo "Sharding ${SOURCE_DATASET} into ${NUM_SHARDS} shards -> ${TARGET_DATASET}..."
python scripts/shard_dataset.py \
    --source-dataset "${SOURCE_DATASET}" \
    --target-dataset "${TARGET_DATASET}" \
    --num-shards "${NUM_SHARDS}"

echo "Sharding complete."
