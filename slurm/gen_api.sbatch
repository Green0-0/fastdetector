#!/bin/bash
#SBATCH --partition=compsci
#SBATCH --job-name=fd_gen_api
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

GEN_CONFIG="config/gen/shard_claude_batch.toml"
SHARD=0

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

echo "Generating shard ${SHARD} from ${GEN_CONFIG} via ${ENGINE}..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "${GEN_CONFIG}" \
    --batch-id "${SHARD}"
