"""Tests for the ``vllm`` tier's environment gate.

The tier gates on a populated engine venv on disk, because that is what the
pipeline actually needs: ``llm_server_context`` launches
``<venv>/bin/vllm`` as a subprocess and nothing in ``src/``, ``scripts/`` or
``tests/`` ever imports the package.

Gating on importability instead forced the tier into ``.vllm``, which holds the
engine but not the project's own dependencies, so the suite could not even
import ``tests/conftest.py`` and the tier never ran to completion.
"""

import importlib.util
import os
from pathlib import Path


def _load_root_conftest():
    """Load ``tests/conftest.py`` by path, under a name of its own.

    A bare ``import conftest`` is ambiguous: ``tests/integration/conftest.py``
    is imported under that same name and wins in ``sys.modules`` during a full
    run, so the import passes in isolation and fails in the suite. Loading by
    path pins the module regardless of collection order.
    """
    path = Path(__file__).resolve().parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("fastdetector_root_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conftest = _load_root_conftest()


def _make_engine_venv(root: Path, executable: bool = True) -> Path:
    """Create ``<root>/bin/vllm`` and return ``root``."""
    binary = root / "bin" / "vllm"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755 if executable else 0o644)
    return root


# --------------------------------------------------------------------------
# Locating the engine venv
# --------------------------------------------------------------------------


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_VENV_PATH", str(tmp_path / "custom"))
    assert conftest._engine_venv_path() == tmp_path / "custom"


def test_globals_toml_is_used_when_no_override(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_VENV_PATH", raising=False)
    monkeypatch.setattr(
        conftest,
        "REPO_ROOT",
        tmp_path,
    )
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "globals.toml").write_text('vllm_venv_path = "engine"\n')
    assert conftest._engine_venv_path() == tmp_path / "engine"


def test_falls_back_to_dot_vllm(monkeypatch, tmp_path):
    """A globals.toml that omits the key (as the committed one does) still resolves."""
    monkeypatch.delenv("VLLM_VENV_PATH", raising=False)
    monkeypatch.setattr(conftest, "REPO_ROOT", tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "globals.toml").write_text('dataset_prefix = "x/y"\n')
    assert conftest._engine_venv_path() == tmp_path / ".vllm"


def test_a_missing_globals_file_does_not_raise(monkeypatch, tmp_path):
    """The gate must degrade to a skip, never a collection error."""
    monkeypatch.delenv("VLLM_VENV_PATH", raising=False)
    monkeypatch.setattr(conftest, "REPO_ROOT", tmp_path)
    assert conftest._engine_venv_path() == tmp_path / ".vllm"


def test_relative_paths_are_anchored_to_the_repo_root(monkeypatch, tmp_path):
    """Otherwise the tier silently depends on the working directory."""
    monkeypatch.setenv("VLLM_VENV_PATH", ".vllm")
    monkeypatch.setattr(conftest, "REPO_ROOT", tmp_path)
    assert conftest._engine_venv_path() == tmp_path / ".vllm"


# --------------------------------------------------------------------------
# Detecting the binary
# --------------------------------------------------------------------------


def test_a_populated_engine_venv_is_detected(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_VENV_PATH", str(_make_engine_venv(tmp_path / "v")))
    assert conftest._engine_binary_available() is True


def test_an_empty_venv_is_not_detected(monkeypatch, tmp_path):
    """A bare `uv venv .vllm` with no engine installed must skip, not fail."""
    (tmp_path / "v" / "bin").mkdir(parents=True)
    monkeypatch.setenv("VLLM_VENV_PATH", str(tmp_path / "v"))
    assert conftest._engine_binary_available() is False


def test_a_non_executable_binary_is_not_detected(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "VLLM_VENV_PATH", str(_make_engine_venv(tmp_path / "v", executable=False))
    )
    assert conftest._engine_binary_available() is False


def test_the_gate_does_not_depend_on_importing_vllm(monkeypatch, tmp_path):
    """The whole point: an engine on disk is enough, no package needed.

    ``vllm`` is not installed in the main venv, so this passing at all is the
    property under test.
    """
    import importlib.util

    assert importlib.util.find_spec("vllm") is None
    monkeypatch.setenv("VLLM_VENV_PATH", str(_make_engine_venv(tmp_path / "v")))
    assert conftest._engine_binary_available() is True


def test_the_committed_repo_state_is_handled(monkeypatch):
    """Against the real repo the gate answers cleanly either way, without raising."""
    monkeypatch.delenv("VLLM_VENV_PATH", raising=False)
    assert isinstance(conftest._engine_binary_available(), bool)
    assert os.path.basename(str(conftest._engine_venv_path())) in (".vllm", "engine")
