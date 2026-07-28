"""Engine subprocess management.

The health-check loop is exercised for real against a stub "engine" binary: a
throwaway script that serves ``/v1/models``. The polling constants are patched
down so the timeout path takes milliseconds instead of twenty minutes.
"""

import json
import os
import socket
import stat
import subprocess
import sys
import textwrap

import pytest

from fastdetector import llm_utils
from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.llm_utils import (
    _resolve_engine_binary,
    _terminate_proc,
    get_free_port,
    get_gpu_count,
    launch_engine_server,
    llm_server_context,
)

STUB_ENGINE = """\
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

argv = sys.argv[1:]
with open(os.environ["STUB_ENGINE_ARGS"], "w") as handle:
    json.dump({"argv": argv, "master_port": os.environ.get("MASTER_PORT")}, handle)

exit_code = os.environ.get("STUB_ENGINE_EXIT")
if exit_code is not None:
    sys.exit(int(exit_code))

if os.environ.get("STUB_ENGINE_HANG"):
    while True:
        time.sleep(3600)

port = int(argv[argv.index("--port") + 1])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"data": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


HTTPServer(("", port), Handler).serve_forever()
"""


@pytest.fixture
def stub_engine(tmp_path, monkeypatch):
    """Install a stub ``vllm`` executable in a fake venv and speed up polling.

    Returns:
        An object with ``venv_path`` and a ``recorded()`` helper that returns
        the argv/env the stub was launched with.
    """
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "vllm"
    script.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(STUB_ENGINE), encoding="utf-8"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    args_file = tmp_path / "args.json"
    monkeypatch.setenv("STUB_ENGINE_ARGS", str(args_file))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(llm_utils, "HEALTH_CHECK_INTERVAL_SECS", 0.05)
    monkeypatch.setattr(llm_utils, "HEALTH_CHECK_MAX_INTERVALS", 200)

    class Stub:
        venv_path = str(tmp_path / "venv")

        @staticmethod
        def recorded() -> dict:
            return json.loads(args_file.read_text(encoding="utf-8"))

    return Stub()


# --------------------------------------------------------------------------
# get_free_port
# --------------------------------------------------------------------------


def test_get_free_port_returns_a_bindable_port():
    port = get_free_port()
    assert isinstance(port, int)
    assert 1024 < port < 65536
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", port))


def test_get_free_port_usually_differs_between_calls():
    ports = {get_free_port() for _ in range(5)}
    assert len(ports) > 1


# --------------------------------------------------------------------------
# get_gpu_count
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("visible", "expected"),
    [("0", 1), ("0,1", 2), ("0,1,2,3", 4), (" 0 , 1 ", 2), ("0,1,", 2)],
)
def test_get_gpu_count_parses_cuda_visible_devices(monkeypatch, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert get_gpu_count() == expected


def test_get_gpu_count_requires_the_variable_to_be_set(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        get_gpu_count()


@pytest.mark.parametrize("visible", ["", " ", ","])
def test_get_gpu_count_rejects_an_empty_device_list(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(RuntimeError, match="No GPUs found"):
        get_gpu_count()


# --------------------------------------------------------------------------
# _resolve_engine_binary
# --------------------------------------------------------------------------


def test_resolve_engine_binary_returns_the_executable_path(stub_engine):
    resolved = _resolve_engine_binary(EngineConfig.VLLM, stub_engine.venv_path)
    assert resolved == os.path.join(stub_engine.venv_path, "bin", "vllm")


def test_resolve_engine_binary_requires_a_venv_path():
    with pytest.raises(RuntimeError, match="venv_path must be provided"):
        _resolve_engine_binary(EngineConfig.VLLM, "")


def test_resolve_engine_binary_reports_a_missing_binary(tmp_path):
    with pytest.raises(RuntimeError, match="not found or not executable"):
        _resolve_engine_binary(EngineConfig.APHRODITE, str(tmp_path))


def test_resolve_engine_binary_rejects_a_non_executable_file(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "vllm"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(RuntimeError, match="not found or not executable"):
        _resolve_engine_binary(EngineConfig.VLLM, str(tmp_path))


def test_resolve_engine_binary_looks_up_the_engine_by_name(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    target = bin_dir / "aphrodite"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    assert _resolve_engine_binary(EngineConfig.APHRODITE, str(tmp_path)).endswith(
        "aphrodite"
    )


# --------------------------------------------------------------------------
# Engine guards
# --------------------------------------------------------------------------


def test_launch_rejects_a_non_local_engine():
    with pytest.raises(ValueError, match="not a local-server engine"):
        launch_engine_server(
            EngineConfig.OAI, "m", 8000, venv_path=".venv", parallelization_type="data"
        )


def test_context_rejects_a_non_local_engine():
    with pytest.raises(ValueError, match="not a local-server engine"):
        with llm_server_context(
            EngineConfig.OAI, "m", venv_path=".venv", parallelization_type="data"
        ):
            pass


# --------------------------------------------------------------------------
# Launching (against the stub engine)
# --------------------------------------------------------------------------


def test_launch_waits_until_the_server_is_healthy(stub_engine, free_port):
    proc = launch_engine_server(
        EngineConfig.VLLM,
        "some/model",
        free_port,
        venv_path=stub_engine.venv_path,
        parallelization_type="data",
    )
    try:
        assert proc.poll() is None
    finally:
        _terminate_proc(proc, EngineConfig.VLLM)


def test_launch_passes_the_configured_engine_flags(stub_engine, free_port):
    proc = launch_engine_server(
        EngineConfig.VLLM,
        "some/model",
        free_port,
        venv_path=stub_engine.venv_path,
        parallelization_type="tensor",
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_seqs=32,
        max_num_batched_tokens=1024,
    )
    try:
        argv = stub_engine.recorded()["argv"]
    finally:
        _terminate_proc(proc, EngineConfig.VLLM)

    assert argv[0] == "serve"
    assert argv[1] == "some/model"
    pairs = {
        flag: argv[index + 1]
        for index, flag in enumerate(argv)
        if flag.startswith("--") and index + 1 < len(argv)
        and not argv[index + 1].startswith("--")
    }
    # CUDA_VISIBLE_DEVICES is "0,1" in the fixture.
    assert pairs["--tensor-parallel-size"] == "2"
    assert pairs["--max-model-len"] == "4096"
    assert pairs["--max-num-seqs"] == "32"
    assert pairs["--max-num-batched-tokens"] == "1024"
    assert pairs["--gpu-memory-utilization"] == "0.75"
    assert pairs["--port"] == str(free_port)
    assert "--disable-uvicorn-access-log" in argv


def test_launch_gives_the_distributed_backend_its_own_port(stub_engine, free_port):
    proc = launch_engine_server(
        EngineConfig.VLLM,
        "some/model",
        free_port,
        venv_path=stub_engine.venv_path,
        parallelization_type="data",
    )
    try:
        recorded = stub_engine.recorded()
    finally:
        _terminate_proc(proc, EngineConfig.VLLM)

    # A MASTER_PORT equal to the API port would make the engine fight itself.
    assert recorded["master_port"] is not None
    assert int(recorded["master_port"]) != free_port


def test_launch_raises_when_the_engine_exits_early(stub_engine, free_port, monkeypatch):
    monkeypatch.setenv("STUB_ENGINE_EXIT", "3")
    with pytest.raises(RuntimeError, match="exited with code 3"):
        launch_engine_server(
            EngineConfig.VLLM,
            "some/model",
            free_port,
            venv_path=stub_engine.venv_path,
            parallelization_type="data",
        )


def test_launch_times_out_and_cleans_up(stub_engine, free_port, monkeypatch):
    monkeypatch.setenv("STUB_ENGINE_HANG", "1")
    monkeypatch.setattr(llm_utils, "HEALTH_CHECK_MAX_INTERVALS", 3)

    with pytest.raises(RuntimeError, match="failed to start within the timeout"):
        launch_engine_server(
            EngineConfig.VLLM,
            "some/model",
            free_port,
            venv_path=stub_engine.venv_path,
            parallelization_type="data",
        )

    # The timed-out process must not be left running.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", free_port))


def test_context_yields_the_api_url_and_shuts_the_server_down(stub_engine, free_port):
    import requests

    with llm_server_context(
        EngineConfig.VLLM,
        "some/model",
        venv_path=stub_engine.venv_path,
        parallelization_type="data",
        port=free_port,
    ) as api_url:
        assert api_url == f"http://localhost:{free_port}/v1"
        assert requests.get(f"{api_url}/models", timeout=5).status_code == 200

    with pytest.raises(requests.RequestException):
        requests.get(f"{api_url}/models", timeout=2)


def test_context_shuts_the_server_down_when_the_body_raises(stub_engine, free_port):
    import requests

    with pytest.raises(ValueError, match="boom"):
        with llm_server_context(
            EngineConfig.VLLM,
            "some/model",
            venv_path=stub_engine.venv_path,
            parallelization_type="data",
            port=free_port,
        ):
            raise ValueError("boom")

    with pytest.raises(requests.RequestException):
        requests.get(f"http://localhost:{free_port}/v1/models", timeout=2)


def test_context_picks_a_free_port_when_none_is_given(stub_engine):
    with llm_server_context(
        EngineConfig.VLLM,
        "some/model",
        venv_path=stub_engine.venv_path,
        parallelization_type="data",
    ) as api_url:
        assert api_url.startswith("http://localhost:")
        port = int(api_url.rsplit(":", 1)[1].split("/")[0])
        assert 1024 < port < 65536


def test_context_does_not_leave_a_process_behind_when_launching_fails(
    tmp_path, monkeypatch
):
    # The binary does not exist, so nothing should be spawned or leaked.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="not found or not executable"):
        with llm_server_context(
            EngineConfig.VLLM,
            "some/model",
            venv_path=str(tmp_path),
            parallelization_type="data",
            port=12345,
        ):
            pass


# --------------------------------------------------------------------------
# _terminate_proc
# --------------------------------------------------------------------------


def test_terminate_proc_stops_a_running_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _terminate_proc(proc, EngineConfig.VLLM)
    assert proc.poll() is not None


@pytest.mark.slow
def test_terminate_proc_escalates_to_kill_when_sigterm_is_ignored():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(60)\n",
        ]
    )
    _terminate_proc(proc, EngineConfig.VLLM, timeout=1)
    assert proc.poll() is not None
