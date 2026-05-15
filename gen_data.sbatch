#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=gen-data-%j.out
#SBATCH --error=gen-data-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a5000:4

# ---- Activate project venv ----
source .venv/bin/activate

# ---- CUDA setup ----
export CUDA_HOME=/usr/local/cuda-12.8/
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---- vLLM venv ----
VLLM_VENV_PATH="${VLLM_VENV_PATH:-${PWD}/.vllm}"
VLLM_PYTHON="${VLLM_VENV_PATH}/bin/python"

if [[ ! -x "${VLLM_PYTHON}" ]]; then
  echo "vLLM venv not found: ${VLLM_VENV_PATH}" >&2
  exit 1
fi

# ---- Model & network config ----
export MODEL_NAME="QuantTrio/Qwen3.5-9B-AWQ"
export VLLM_HOST="0.0.0.0"
export VLLM_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

echo "Starting vLLM on port ${VLLM_PORT}..."

# ---- Cleanup trap ----
cleanup_vllm() {
  echo "Shutting down vLLM..."
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT

# ---- Start vLLM in the background ----
"${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 49152 &
VLLM_PID=$!

# ---- Wait for vLLM to become healthy ----
python - <<'PY'
import time, urllib.request, os

port = os.environ["VLLM_PORT"]
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + 600

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print(f"vLLM ready on port {port}")
                raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        time.sleep(2)

raise SystemExit("vLLM did not become ready within 10 minutes.")
PY

if [ $? -ne 0 ]; then
  echo "vLLM failed to start."
  exit 1
fi

# ---- Pass the endpoint to main.py via environment ----
HOSTNAME=$(hostname)
export VLLM_API_URL="http://${HOSTNAME}:${VLLM_PORT}/v1"
echo "vLLM endpoint: ${VLLM_API_URL}"

# ---- Run the dataset generation ----
echo "Starting dataset generation..."
python generate_llm_data.py

echo "Dataset generation complete."
