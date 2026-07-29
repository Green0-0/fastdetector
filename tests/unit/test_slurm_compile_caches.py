"""GPU job scripts must keep their compile caches off NFS.

Nothing set ``TRITON_CACHE_DIR``, so Triton used ``$HOME/.triton``. On this
cluster ``$HOME`` is NFS, and several array tasks starting engines at once
compile the same kernels into one shared directory, race, and get ``ESTALE``
back. It surfaces as ``Triton compilation failed`` / ``OSError: [Errno 116]
Stale file handle``, which reads like a CUDA problem rather than a
shared-filesystem one. ``$HOME`` is also quota-limited, so filling it with
compile artifacts is undesirable regardless.

The set of scripts that need this is derived from ``--gres=gpu`` rather than
hardcoded, so a new GPU stage is covered the moment it is added.

Textual checks, so they run in the default tier without Slurm.
"""

import re

import pytest

#: Caches every script running torch on a GPU must redirect.
TORCH_CACHE_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")

#: Additionally required where an engine is launched.
ENGINE_CACHE_VAR = "VLLM_CACHE_ROOT"


def _job_scripts(repo_root):
    return sorted((repo_root / "slurm").rglob("*.sbatch"))


def _requests_gpu(text: str) -> bool:
    return any(
        line.strip().startswith("#SBATCH") and "--gres=gpu" in line
        for line in text.splitlines()
    )


def _launches_engine(text: str) -> bool:
    """True if the script expects an engine binary, i.e. vLLM will run."""
    return "bin/vllm" in text


def _gpu_scripts(repo_root):
    return [p for p in _job_scripts(repo_root) if _requests_gpu(p.read_text())]


def _exported_value(text: str, var: str):
    """Return the value assigned by ``export VAR=...``, or None."""
    match = re.search(rf'^export {var}="?([^"\n]+)"?', text, re.M)
    return match.group(1) if match else None


def test_some_scripts_request_a_gpu(repo_root):
    """Guard against the detection silently matching nothing."""
    assert _gpu_scripts(repo_root)


@pytest.mark.parametrize("var", TORCH_CACHE_VARS)
def test_gpu_scripts_redirect_the_compile_caches(repo_root, var):
    missing = [p.name for p in _gpu_scripts(repo_root) if not _exported_value(p.read_text(), var)]
    assert not missing, f"{var} is not exported in: {missing}"


@pytest.mark.parametrize("var", TORCH_CACHE_VARS)
def test_the_caches_point_at_node_local_disk(repo_root, var):
    """A path under $HOME is the bug; /tmp is node-local and unshared."""
    offenders = []
    for path in _gpu_scripts(repo_root):
        value = _exported_value(path.read_text(), var) or ""
        if "$HOME" in value or value.startswith("~"):
            offenders.append(f"{path.name}: {var}={value}")
    assert not offenders, "compile caches must not live on NFS:\n  " + "\n  ".join(offenders)


def test_each_job_gets_its_own_cache_root(repo_root):
    """A shared directory is the whole failure: concurrent tasks race in it.

    ``SLURM_JOB_ID`` is distinct per array task, so it separates them.
    """
    offenders = []
    for path in _gpu_scripts(repo_root):
        text = path.read_text()
        for var in TORCH_CACHE_VARS:
            value = _exported_value(text, var) or ""
            resolved = value
            if "FD_CACHE_ROOT" in value:
                root = re.search(r'^FD_CACHE_ROOT="?([^"\n]+)"?', text, re.M)
                resolved = root.group(1) if root else ""
            if "SLURM_JOB_ID" not in resolved and "$$" not in resolved:
                offenders.append(f"{path.name}: {var}={value}")
    assert not offenders, (
        "cache roots must be per-job or tasks race each other:\n  " + "\n  ".join(offenders)
    )


def test_the_cache_directories_are_created(repo_root):
    """Triton will not create a missing TRITON_CACHE_DIR for you."""
    missing = [
        p.name
        for p in _gpu_scripts(repo_root)
        if not re.search(r"^mkdir -p .*TRITON_CACHE_DIR", p.read_text(), re.M)
    ]
    assert not missing, f"no mkdir for the cache dirs in: {missing}"


def test_the_cache_root_is_cleaned_up(repo_root):
    """Without a trap, /tmp fills up over a run of array jobs."""
    missing = [
        p.name
        for p in _gpu_scripts(repo_root)
        if not re.search(r"^trap .*rm -rf.*EXIT", p.read_text(), re.M)
    ]
    assert not missing, f"no EXIT trap cleaning the cache root in: {missing}"


def test_engine_scripts_also_redirect_the_vllm_cache(repo_root):
    missing = [
        p.name
        for p in _gpu_scripts(repo_root)
        if _launches_engine(p.read_text())
        and not _exported_value(p.read_text(), ENGINE_CACHE_VAR)
    ]
    assert not missing, f"{ENGINE_CACHE_VAR} is not exported in: {missing}"


def test_cpu_only_scripts_are_left_alone(repo_root):
    """No reason to add compile caches to stages that never load torch."""
    cpu_only = [p for p in _job_scripts(repo_root) if not _requests_gpu(p.read_text())]
    assert cpu_only, "expected at least one CPU-only stage"
    for path in cpu_only:
        assert "TRITON_CACHE_DIR" not in path.read_text()
