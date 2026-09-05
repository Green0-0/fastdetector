#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --job-name=fd_gen_api
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=8-9

set -euo pipefail

DATASET_KIND="${DATASET_KIND:-train}"
SHARD="${SLURM_ARRAY_TASK_ID:-8}"
case "${DATASET_KIND}" in
  train|val|test) ;;
  *) echo "DATASET_KIND must be train, val, or test." >&2; exit 1 ;;
esac
GEN_CONFIG="config/gen/${DATASET_KIND}/shard_${SHARD}.toml"

source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

read_key() {
  sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*\"\([^\"]*\)\".*/\1/p" "${GEN_CONFIG}" | head -1
}

ENGINE=$(read_key engine)
case "${ENGINE}" in
  oai|anthropic|anthropic_aws) ;;
  vllm|aphrodite)
    echo "${GEN_CONFIG} uses the ${ENGINE} engine; submit it with gen.sbatch (needs a GPU)." >&2
    exit 1
    ;;
  *)
    echo "Could not read a hosted-API engine from ${GEN_CONFIG} (got '${ENGINE}')." >&2
    exit 1
    ;;
esac

API_KEY_ENV=$(read_key api_key_env)
if [[ -n "${API_KEY_ENV}" && -z "${!API_KEY_ENV:-}" ]]; then
  echo "${API_KEY_ENV} is unset, but ${GEN_CONFIG} names it as the API key." >&2
  exit 1
fi
if [[ "${ENGINE}" == "anthropic_aws" ]]; then
  : "${ANTHROPIC_AWS_WORKSPACE_ID:?must be set for Claude Platform on AWS}"
fi

echo "Generating ${DATASET_KIND} shard ${SHARD} from ${GEN_CONFIG} via ${ENGINE}..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "${GEN_CONFIG}" \
    --batch-id "${SHARD}"
