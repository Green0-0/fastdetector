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

# ---- Compile caches on node-local disk ----
# Triton defaults to $HOME/.triton. $HOME is NFS here, and concurrent engines
# compiling the same kernels into one shared directory race each other; NFS
# answers with ESTALE ("Stale file handle"), which surfaces as a Triton
# compilation failure and kills the worker. $HOME is also quota-limited, so
# filling it with compile artifacts is undesirable regardless.
# SLURM_JOB_ID is distinct per array task, so each task gets its own root.
FD_CACHE_ROOT="/tmp/fd_${SLURM_JOB_ID:-$$}"
export TRITON_CACHE_DIR="${FD_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${FD_CACHE_ROOT}/inductor"
export VLLM_CACHE_ROOT="${FD_CACHE_ROOT}/vllm"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
trap 'rm -rf "${FD_CACHE_ROOT}"' EXIT

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
# Runs from the MAIN venv, and never writes to the engine venv.
#
# The engine venv lives on NFS and running vLLM engines execute out of it.
# Rewriting it mid-run invalidates their open file handles, which surfaces as
# ESTALE inside Triton compilation and kills the engine -- so a routine test
# run would silently destroy unrelated multi-hour production jobs.
#
# Nothing here needs to be installed anyway: the pipeline does not import
# vllm, it launches <VLLM_VENV_PATH>/bin/vllm as a subprocess. The tier
# therefore gates on that binary existing, not on a package being importable.
VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
export VLLM_VENV_PATH
if [[ -x "${VLLM_VENV_PATH}/bin/vllm" ]]; then
  echo
  echo "=============================================================="
  echo "vLLM tier (boots the configured engine end to end)"
  echo "=============================================================="
  pytest -q -m vllm || status=1
else
  echo
  echo "No engine binary at ${VLLM_VENV_PATH}/bin/vllm; skipping the vllm tier." >&2
fi

echo
if [[ ${status} -eq 0 ]]; then
  echo "All selected tiers passed."
else
  echo "One or more tiers FAILED - see above." >&2
fi
exit ${status}
