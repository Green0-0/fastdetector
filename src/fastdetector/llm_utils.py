import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

import requests

from fastdetector.frontend.engine_config import EngineConfig

HEALTH_CHECK_INTERVAL_SECS = 2
HEALTH_CHECK_MAX_INTERVALS = 600

HEALTH_CHECK_PROGRESS_INTERVAL_SECS = 30


def get_free_port() -> int:
    """Find and return a free port on localhost.

    Uses an ephemeral bind to let the OS assign a free port, then immediately
    closes the socket. There is an inherent TOCTOU race (another process may
    grab the port between this call and its use), but for local LLM servers
    this is acceptable.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_gpu_count() -> int:
    """Autodetect the number of available GPUs from CUDA_VISIBLE_DEVICES.

    Returns:
        Number of available GPUs.

    Raises:
        RuntimeError: if CUDA_VISIBLE_DEVICES is unset or empty.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
        if len(devices) == 0:
            raise RuntimeError("No GPUs found! CUDA_VISIBLE_DEVICES=" + cuda_visible)
        return len(devices)
    raise RuntimeError("No GPUs found! CUDA_VISIBLE_DEVICES not set.")


def _resolve_engine_binary(engine: EngineConfig, venv_path: str) -> str:
    """Resolve the engine binary path and return bin_path.

    Args:
        engine: The engine to resolve the binary for.
        venv_path: The path to the virtual environment.

    Returns:
        The path to the engine binary.

    Raises:
        RuntimeError: if the binary is missing.
    """
    if not venv_path:
        raise RuntimeError(
            f"venv_path must be provided in the global config for the {engine.value} engine."
        )
    bin_name = engine.value
    bin_path = os.path.join(venv_path, "bin", bin_name)

    if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
        raise RuntimeError(
            f"{engine.value} executable not found or not executable at: {bin_path}"
        )

    return bin_path


def launch_engine_server(
    engine: EngineConfig,
    model_name: str,
    port: int,
    venv_path: str,
    parallelization_type: str,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 16000,
) -> subprocess.Popen:
    """Launch the LLM server with data-parallel size = GPU count.

    The subprocess's stdout and stderr are streamed to the parent process's
    stdout/stderr so that engine startup logs (model loading progress, CUDA
    errors, etc.) are visible.

    Args:
        engine: Engine for the backend. Must be a local-server engine
            (vLLM or Aphrodite).
        model_name: HuggingFace model ID or local path.
        port: Port for the OpenAI-compatible API server.
        venv_path: Path to the python virtual environment containing the engine executable.
        parallelization_type: Parallelization strategy ("data", "tensor", "pipeline").
        gpu_memory_utilization: Fraction of GPU memory to use (0–1).
        max_model_len: Maximum model context length.

    Returns:
        The subprocess.Popen handle once the server is healthy.

    Raises:
        ValueError: if the engine is not a local-server engine.
        RuntimeError: if the server fails to become healthy within the timeout
            (currently 20 minutes). The subprocess is terminated before raising.
    """
    if not engine.is_local_server:
        raise ValueError(
            f"{engine} is not a local-server engine. "
            f"Only {EngineConfig.VLLM.value} and {EngineConfig.APHRODITE.value} can be launched."
        )

    gpu_count = get_gpu_count()
    bin_path = _resolve_engine_binary(engine, venv_path)

    cmd_prefix = [bin_path, "serve", model_name]

    cmd = cmd_prefix + [
        "--port", str(port),
        f"--{parallelization_type}-parallel-size", str(gpu_count),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", "256",
        "--max-num-batched-tokens", "2048",
        "--disable-uvicorn-access-log",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    dist_port = port
    while dist_port == port:
        dist_port = get_free_port()
        if dist_port == port:
            time.sleep(0.01)

    env = os.environ.copy()
    env["MASTER_PORT"] = str(dist_port)

    print(f"\nLaunching {engine} server...")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(
        f"Waiting for {engine} server to start "
        f"(timeout: {HEALTH_CHECK_MAX_INTERVALS * HEALTH_CHECK_INTERVAL_SECS}s)..."
    )
    health_url = f"http://localhost:{port}/v1/models"

    last_progress_print = time.time()
    for i in range(HEALTH_CHECK_MAX_INTERVALS):
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print(f"{engine} server is ready.\n")
                return proc
        except Exception:
            pass

        now = time.time()
        if now - last_progress_print >= HEALTH_CHECK_PROGRESS_INTERVAL_SECS:
            elapsed = i * HEALTH_CHECK_INTERVAL_SECS
            remaining = (HEALTH_CHECK_MAX_INTERVALS - i) * HEALTH_CHECK_INTERVAL_SECS
            print(
                f"  Still waiting for {engine} server... "
                f"{elapsed}s elapsed, {remaining}s remaining"
            )
            last_progress_print = now

        time.sleep(HEALTH_CHECK_INTERVAL_SECS)

    print(
        f"{engine} server failed to start within the timeout "
        f"({HEALTH_CHECK_MAX_INTERVALS * HEALTH_CHECK_INTERVAL_SECS}s). "
        f"Attempting to shut it down..."
    )
    _terminate_proc(proc, engine)
    raise RuntimeError(
        f"{engine} server failed to start within the timeout. "
        f"Check the engine logs above for details."
    )


def _terminate_proc(proc: subprocess.Popen, engine: EngineConfig, timeout: int = 60) -> None:
    """Terminate a subprocess gracefully, escalating to kill if needed.

    Args:
        proc: The subprocess handle to terminate.
        engine: Engine configuration describing the server process.
        timeout: Time in seconds to wait for termination before sending SIGKILL.

    Returns:
        None.
    """
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{engine.value} server did not terminate, killing it...")
        proc.kill()
        proc.wait()


@contextmanager
def llm_server_context(
    engine: EngineConfig,
    model_name: str,
    venv_path: str,
    parallelization_type: str,
    port: int | None = None,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 16000,
):
    """Context manager to launch and clean up an LLM server.

    Launches the engine server on a free port (or *port* if given), yields
    the API base URL ("http://localhost:{port}/v1"), and terminates the
    server on context exit (including on exception).

    Args:
        engine: Engine for the backend. Must be a local-server engine
            (vLLM or Aphrodite).
        model_name: HuggingFace model ID or local path.
        venv_path: Path to the python virtual environment containing the engine executable.
        parallelization_type: Parallelization strategy ("data", "tensor", "pipeline").
        port: Port for the API server. If None, a free port is chosen.
        gpu_memory_utilization: Fraction of GPU memory to use (0–1).
        max_model_len: Maximum model context length.

    Yields:
        The API base URL string.

    Raises:
        ValueError: if the engine is not a local-server engine.
        RuntimeError: if the server fails to start within the timeout.
    """
    if not engine.is_local_server:
        raise ValueError(
            f"{engine} is not a local-server engine. "
            f"Only {EngineConfig.VLLM.value} and {EngineConfig.APHRODITE.value} can be launched."
        )

    if port is None:
        port = get_free_port()

    proc = None
    try:
        proc = launch_engine_server(
            engine, 
            model_name, 
            port,
            venv_path=venv_path,
            parallelization_type=parallelization_type,
            gpu_memory_utilization=gpu_memory_utilization, 
            max_model_len=max_model_len,
        )
        yield f"http://localhost:{port}/v1"
    finally:
        if proc is not None:
            print(f"Shutting down {engine.value} server...")
            _terminate_proc(proc, engine)
            print(f"{engine.value} server shutdown complete.")
