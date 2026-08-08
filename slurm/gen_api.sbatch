#!/bin/bash
# Generation against a hosted API (no local engine, no GPU).
#
# Offline batch jobs have a 24h completion window per chat turn, and prompts
# with two turns serialise that wait twice, so the walltime here is deliberately
# generous. The run is resumable: submitted batch IDs are persisted under
# batch_state_dir, so a requeued job resumes polling instead of resubmitting
# (and re-paying for) work already in flight.
#SBATCH --partition=compsci
#SBATCH --job-name=fd_gen_api
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-8

set -euo pipefail

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

GEN_CONFIG="config/gen/shard_${SLURM_ARRAY_TASK_ID:-0}.toml"

if [[ ! -f "${GEN_CONFIG}" ]]; then
  echo "No gen config at ${GEN_CONFIG}." >&2
  exit 1
fi

ENGINE=$(sed -n 's/^[[:space:]]*engine[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${GEN_CONFIG}" | head -1)
case "${ENGINE}" in
  oai|azure_oai|anthropic|anthropic_aws) ;;
  vllm|aphrodite)
    echo "${GEN_CONFIG} uses the ${ENGINE} engine; submit it with gen.sbatch (needs a GPU)." >&2
    exit 1
    ;;
  *)
    echo "Could not read a hosted-API engine from ${GEN_CONFIG} (got '${ENGINE}')." >&2
    exit 1
    ;;
esac

# Fail fast on missing credentials rather than after building every request.
case "${ENGINE}" in
  anthropic_aws)
    : "${ANTHROPIC_AWS_WORKSPACE_ID:?must be set for Claude Platform on AWS}"
    : "${AWS_REGION:?must be set for Claude Platform on AWS}"
    ;;
esac

echo "Generating shard ${SLURM_ARRAY_TASK_ID:-0} via ${ENGINE}..."
python scripts/gen.py \
    --globals-config config/globals.toml \
    --gen-config "${GEN_CONFIG}" \
    --batch-id "${SLURM_ARRAY_TASK_ID:-0}"

echo "Generation complete for shard ${SLURM_ARRAY_TASK_ID:-0}."
