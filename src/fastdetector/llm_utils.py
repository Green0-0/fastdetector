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

def launch_engine_server(engine: str, model_name: str, port: int, max_logprobs: int = 10, gpu_memory_utilization: float = 0.85, max_model_len: int = 16000, ) -> subprocess.Popen:
    """Launch the LLM server with pipeline parallel size equal to the number of GPUs."""
    gpu_count = get_gpu_count()
    
    if engine.lower() == "vllm":
        venv_path = os.environ.get("VLLM_VENV_PATH")
        if not venv_path:
            raise RuntimeError("VLLM_VENV_PATH environment variable is not set. Please set VLLM_VENV_PATH to the path of the virtual environment containing the vLLM installation.")
        bin_path = os.path.join(venv_path, "bin", "vllm")
        cmd_prefix = [bin_path, "serve", model_name]
    elif engine.lower() == "aphrodite":
        venv_path = os.environ.get("APHRODITE_VENV_PATH", ".aphrodite")
        bin_path = os.path.join(venv_path, "bin", "aphrodite")
        cmd_prefix = [bin_path, "run", model_name]
    else:
        raise ValueError(f"Unsupported LLM engine: {engine}")

    if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
        raise RuntimeError(f"{engine} executable not found or not executable at: {bin_path}")

    cmd = cmd_prefix + [
        "--port", str(port),
        "--data-parallel-size", str(gpu_count),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "2048",
        "--disable-uvicorn-access-log",
        "--max-logprobs", str(max_logprobs),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    while (dist_port := get_free_port()) == port: pass
    env = os.environ.copy()
    env["MASTER_PORT"] = str(dist_port)

    print(f"\nLaunching {engine} server...")
    print(" ".join(cmd))

    proc = subprocess.Popen(cmd, env=env)

    print(f"Waiting for {engine} server to start (this may take a few minutes)...")
    health_url = f"http://localhost:{port}/v1/models"

    for _ in range(600):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print(f"{engine} server is ready.\n")
                return proc
        except Exception:
            pass
        time.sleep(2)

    print(f"{engine} server failed to start within the timeout. Attempting to shut it down...")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print(f"{engine} server did not terminate, killing it...")
        proc.kill()
        proc.wait()
    raise RuntimeError(f"{engine} server failed to start within the timeout.")

@contextmanager
def llm_server_context(engine: str, model_name: str, port: int | None = None, max_logprobs: int = 10, gpu_memory_utilization: float = 0.85, max_model_len: int = 16000):
    """Context manager to launch and clean up an LLM server."""
    if port is None:
        port = get_free_port()

    proc = None
    try:
        if engine.lower() in ["vllm", "aphrodite"]:
            proc = launch_engine_server(engine, model_name, port, max_logprobs, gpu_memory_utilization, max_model_len)
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