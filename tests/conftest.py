"""Shared fixtures and tier gating for the FastDetector test suite.

Two independent mechanisms decide whether a test runs:

* **Markers** declare *intent* — "this needs a GPU", "this downloads data".
  ``pyproject.toml`` deselects the expensive ones by default.
* **Capability checks** (``pytest_runtest_setup`` below) declare *environment* —
  a ``gpu``-marked test that is explicitly selected on a CPU-only box skips with
  a reason instead of erroring.

Everything in the default tier runs offline: the model fixtures below build
randomly-initialised checkpoints in-process instead of downloading them.
"""

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

# Must be set before anything imports pyplot (fastdetector.visualization does).
os.environ.setdefault("MPLBACKEND", "Agg")
# Keep tokenizers from forking after we use threads in the scorer tests.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).parent / "data"

#: Vocabulary size shared by every synthetic model/tokenizer fixture.
TINY_VOCAB_SIZE = 64


# --------------------------------------------------------------------------
# Tier gating
# --------------------------------------------------------------------------


def _cuda_available() -> bool:
    """Return True if torch reports at least one usable CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _network_disabled() -> str | None:
    """Return a skip reason if network access has been switched off, else None."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return "HF_HUB_OFFLINE=1"
    if os.environ.get("FASTDETECTOR_TEST_OFFLINE") == "1":
        return "FASTDETECTOR_TEST_OFFLINE=1"
    return None


def _engine_venv_path() -> Path:
    """Return the engine venv the pipeline would launch its server from.

    Resolution order matches the pipeline's own: the ``VLLM_VENV_PATH``
    override, then ``vllm_venv_path`` in globals.toml, then the ``.vllm``
    default. Relative paths are anchored to the repository root so the gate
    does not depend on the working directory.

    Returns:
        Absolute path to the engine venv.
    """
    venv = os.environ.get("VLLM_VENV_PATH")
    if not venv:
        try:
            from fastdetector.frontend.toml_loader import load_toml

            venv = load_toml(str(REPO_ROOT / "config" / "globals.toml")).get(
                "vllm_venv_path"
            )
        except Exception:
            venv = None
    path = Path(venv or ".vllm")
    return path if path.is_absolute() else REPO_ROOT / path


def _engine_binary_available() -> bool:
    """Return True if a populated engine venv exists on disk.

    The pipeline never imports vllm -- ``llm_server_context`` launches
    ``<venv>/bin/vllm`` as a subprocess -- so an executable binary, not an
    importable package, is what the ``vllm`` tier actually requires.
    """
    return os.access(_engine_venv_path() / "bin" / "vllm", os.X_OK)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests whose environment requirement is not met.

    Runs per test (after marker selection), so the checks stay lazy: a plain
    ``pytest`` run never imports torch just to decide it has no GPU tests.

    Args:
        item: The test about to run.
    """
    if "gpu" in item.keywords and not _cuda_available():
        pytest.skip("no CUDA device visible")
    if "network" in item.keywords:
        reason = _network_disabled()
        if reason:
            pytest.skip(f"network tests disabled ({reason})")
    if "vllm" in item.keywords and not _engine_binary_available():
        pytest.skip(f"no engine binary at {_engine_venv_path() / 'bin' / 'vllm'}")


# --------------------------------------------------------------------------
# Paths and environment
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Absolute path to the committed test-fixture directory."""
    return DATA_DIR


@pytest.fixture
def require_vram():
    """Return a callable that skips the test unless N GiB of free VRAM exists.

    Usage inside a ``gpu``-marked test::

        def test_thing(require_vram):
            require_vram(24)
    """

    def _require(gb: float) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device visible")
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024**3
        if free_gb < gb:
            pytest.skip(f"needs {gb} GiB free VRAM, only {free_gb:.1f} GiB free")

    return _require


@pytest.fixture
def cuda_device() -> str:
    """The CUDA device GPU-tier tests should run on."""
    return os.environ.get("FASTDETECTOR_TEST_CUDA_DEVICE", "cuda:0")


@pytest.fixture(scope="session")
def hub_model_id() -> str:
    """Small public checkpoint used by the ``network`` model tier."""
    return os.environ.get(
        "FASTDETECTOR_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )


# --------------------------------------------------------------------------
# Offline model / tokenizer fixtures
# --------------------------------------------------------------------------


def _build_causal_lm(vocab_size: int = TINY_VOCAB_SIZE, seed: int = 0):
    """Build a tiny randomly-initialised Llama in eval mode, with no downloads.

    Args:
        vocab_size: Output vocabulary size.
        seed: Torch manual seed, so two models can be made deliberately
            different (or identical) without touching the Hub.

    Returns:
        A ``LlamaForCausalLM`` on CPU in float32, in eval mode.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture(scope="session")
def make_causal_lm():
    """Factory fixture returning :func:`_build_causal_lm`."""
    return _build_causal_lm


@pytest.fixture(scope="session")
def tiny_lm():
    """A single tiny CausalLM (seed 0), shared across the session."""
    return _build_causal_lm(seed=0)


@pytest.fixture(scope="session")
def tiny_lm_pair():
    """Two distinct tiny CausalLMs that share a vocabulary (Binoculars setup)."""
    return [_build_causal_lm(seed=0), _build_causal_lm(seed=1)]


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """A whitespace word-level tokenizer aligned with the tiny model vocab.

    Token ``w<i>`` maps to id ``i`` for ``i`` in ``[0, 60)``, which makes the
    expected token ids of a test string obvious by inspection.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(TINY_VOCAB_SIZE - 4)}
    vocab["[UNK]"] = TINY_VOCAB_SIZE - 4
    vocab["[PAD]"] = TINY_VOCAB_SIZE - 3
    vocab["[CLS]"] = TINY_VOCAB_SIZE - 2
    vocab["[SEP]"] = TINY_VOCAB_SIZE - 1

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture(scope="session")
def tiny_sequence_classifier():
    """Factory building a tiny BERT sequence classifier with N labels, offline."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    def _build(num_labels: int = 5, seed: int = 0):
        torch.manual_seed(seed)
        config = BertConfig(
            vocab_size=TINY_VOCAB_SIZE,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=256,
            num_labels=num_labels,
        )
        model = BertForSequenceClassification(config)
        model.eval()
        return model

    return _build


# --------------------------------------------------------------------------
# Fake OpenAI-compatible server
# --------------------------------------------------------------------------


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat-completions handler."""

    def do_POST(self) -> None:  # noqa: N802 (name fixed by BaseHTTPRequestHandler)
        """Record the request body and reply with the responder's payload."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        with self.server.lock:
            self.server.received.append(payload)
        status, body = self.server.responder(payload)
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        """Answer the ``/v1/models`` health probe."""
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        """Silence the default stderr access log."""


def _default_responder(payload: dict) -> tuple[int, dict]:
    """Echo the last user message back as the assistant response.

    Args:
        payload: The decoded chat-completions request body.

    Returns:
        Tuple of (HTTP status, response body).
    """
    messages = payload.get("messages") or []
    last_user = ""
    for message in messages:
        if message.get("role") == "user":
            last_user = message.get("content", "")
    return 200, {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 0,
        "model": payload.get("model", "fake-model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"echo:{last_user}"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


class FakeOpenAIServer:
    """A localhost OpenAI-compatible endpoint for exercising the real client.

    Attributes:
        url: Base URL to hand to ``batch_generate`` (``http://127.0.0.1:P/v1``).
        received: Every decoded request body, in arrival order.
    """

    def __init__(self) -> None:
        """Start the background HTTP server on an ephemeral port."""
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
        self._httpd.responder = _default_responder
        self._httpd.received = []
        self._httpd.lock = threading.Lock()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """The port the server is listening on."""
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        """OpenAI-compatible base URL."""
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def received(self) -> list[dict]:
        """Request bodies received so far."""
        with self._httpd.lock:
            return list(self._httpd.received)

    def set_responder(self, responder) -> None:
        """Install a ``payload -> (status, body)`` callable.

        Args:
            responder: Function producing the reply for each request.
        """
        self._httpd.responder = responder

    def close(self) -> None:
        """Shut the server down and join its thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def fake_openai_server():
    """A running :class:`FakeOpenAIServer`, torn down after the test."""
    server = FakeOpenAIServer()
    try:
        yield server
    finally:
        server.close()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


@pytest.fixture
def free_port() -> int:
    """An ephemeral port number that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
