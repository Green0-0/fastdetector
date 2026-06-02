import os
import socket
import subprocess
import time
from contextlib import contextmanager
import requests

def get_free_port() -> int:
    """Find and return a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def get_gpu_count() -> int:
    """Autodetect the number of available GPUs from CUDA_VISIBLE_DEVICES."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
        if len(devices) == 0:
            raise RuntimeError("No GPUs found! CUDA_VISIBLE_DEVICES=" + cuda_visible)
        return len(devices)
    raise RuntimeError("No GPUs found! CUDA_VISIBLE_DEVICES not set.")

def launch_vllm_server(model_name: str, port: int) -> subprocess.Popen:
    """Launch the vLLM server with pipeline parallel size equal to the number of GPUs."""
    gpu_count = get_gpu_count()
    vllm_venv = os.environ.get("VLLM_VENV_PATH")
    if not vllm_venv:
        raise RuntimeError("VLLM_VENV_PATH environment variable is not set. Please set VLLM_VENV_PATH to the path of the virtual environment containing the vLLM installation.")

    vllm_bin = os.path.join(vllm_venv, "bin", "vllm")
    if not os.path.isfile(vllm_bin) or not os.access(vllm_bin, os.X_OK):
        raise RuntimeError(f"vLLM executable not found or not executable at: {vllm_bin}")

    cmd = [
        vllm_bin, "serve", model_name,
        "--port", str(port),
        "--data-parallel-size", str(gpu_count),
        "--max-model-len", "16000",
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "2048",
        "--disable-uvicorn-access-log",
        "--max-logprobs", "100",
    ]

    # Find a free port for PyTorch Distributed master process to avoid collisions
    while (dist_port := get_free_port()) == port: pass
    env = os.environ.copy()
    env["MASTER_PORT"] = str(dist_port)

    print("\nLaunching vLLM server...")
    print(" ".join(cmd))

    proc = subprocess.Popen(cmd, env=env)

    print("Waiting for vLLM server to start (this may take a few minutes)...")
    health_url = f"http://localhost:{port}/v1/models"

    for _ in range(600):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("vLLM server is ready.\n")
                return proc
        except Exception:
            pass
        time.sleep(2)

    print("vLLM server failed to start within the timeout. Attempting to shut it down...")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print(f"vLLM server did not terminate, killing it...")
        proc.kill()
        proc.wait()
    raise RuntimeError("vLLM server failed to start within the timeout.")

@contextmanager
def llm_server_context(engine: str, model_name: str, port: int | None = None):
    """Context manager to launch and clean up an LLM server."""
    if port is None:
        port = get_free_port()

    proc = None
    try:
        if engine.lower() == "vllm":
            proc = launch_vllm_server(model_name, port)
        else:
            raise ValueError(f"Unsupported LLM engine: {engine}")
        yield f"http://localhost:{port}/v1"
    finally:
        if proc is not None:
            print(f"Shutting down {engine} server...")
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(f"{engine} server did not terminate, killing it...")
                proc.kill()
                proc.wait()
            print(f"{engine} server shutdown complete.")