#!/bin/bash
#SBATCH --job-name=fastdetector_tests
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a5000:1

set -uo pipefail

# ---- Activate project venv ----
source .venv/bin/activate

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-13.0/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTHONUNBUFFERED=1

# Share one Hub cache between runs so the network tier downloads once.
export HF_HOME="${HF_HOME:-/usr/project/xtmp/fastdetector_hf_cache}"

# Fraction of VRAM a stage is allowed to peak at during the preflight.
export FASTDETECTOR_TEST_VRAM_BUDGET="${FASTDETECTOR_TEST_VRAM_BUDGET:-0.9}"

status=0

echo "=============================================================="
echo "Default tier (fast, offline)"
echo "=============================================================="
pytest -q || status=1

echo
echo "=============================================================="
echo "Slow tier"
echo "=============================================================="
pytest -q -m "slow and not network and not gpu and not vllm" || status=1

echo
echo "=============================================================="
echo "Network tier (real checkpoints from the Hub)"
echo "=============================================================="
pytest -q -m "network and not gpu and not vllm" || status=1

echo
echo "=============================================================="
echo "GPU preflight (peak VRAM against the real configs)"
echo "=============================================================="
# -s so the "[preflight] ... peak X GiB / Y GiB" lines reach the job log even
# when the assertions pass.
pytest -m "gpu and not vllm" -s || status=1

# ---- vLLM tier ----
VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
if [[ -d "${VLLM_VENV_PATH}" ]]; then
  echo
  echo "=============================================================="
  echo "vLLM tier (boots the configured engine end to end)"
  echo "=============================================================="
  deactivate
  source "${VLLM_VENV_PATH}/bin/activate"
  # pytest lives in the main venv; install it here too if it is missing.
  python -c "import pytest" 2>/dev/null || uv pip install "pytest>=8.0"
  python -m pytest -q -m vllm || status=1
else
  echo
  echo "vLLM venv not found at ${VLLM_VENV_PATH}; skipping the vllm tier." >&2
fi

echo
if [[ ${status} -eq 0 ]]; then
  echo "All selected tiers passed."
else
  echo "One or more tiers FAILED - see above." >&2
fi
exit ${status}
