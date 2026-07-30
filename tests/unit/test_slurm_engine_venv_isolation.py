import re

import pytest

#: Anything that mutates a venv in place.
INSTALL_COMMANDS = (
    re.compile(r"\buv\s+pip\s+install\b"),
    re.compile(r"\bpip3?\s+install\b"),
    re.compile(r"\buv\s+sync\b"),
    re.compile(r"\buv\s+add\b"),
)

#: Sourcing the engine venv's activate script, however the path is spelled.
ACTIVATE_ENGINE_VENV = re.compile(
    r"(source|\.)\s+[\"']?\S*(VLLM_VENV_PATH|APHRODITE_VENV_PATH|\.vllm|\.aphrodite)\S*/bin/activate"
)


def _job_scripts(repo_root):
    return sorted((repo_root / "slurm").rglob("*.sbatch"))


def _code_lines(text):
    """Yield ``(number, code)`` with comments stripped and blanks skipped."""
    for number, line in enumerate(text.splitlines(), start=1):
        code = line.split("#", 1)[0].strip()
        if code:
            yield number, code


def test_there_are_job_scripts_to_check(repo_root):
    assert _job_scripts(repo_root)


def test_no_job_script_activates_the_engine_venv(repo_root):
    """Every tier runs from the main venv; the engine is a subprocess, not an interpreter."""
    offenders = [
        f"{path.name}:{number}: {code}"
        for path in _job_scripts(repo_root)
        for number, code in _code_lines(path.read_text())
        if ACTIVATE_ENGINE_VENV.search(code)
    ]
    assert not offenders, (
        "job scripts must not run out of the engine venv:\n  " + "\n  ".join(offenders)
    )


def test_no_job_script_installs_packages(repo_root):
    """Installing at job time races other jobs and mutates shared NFS state.

    Provisioning belongs in setup (README.md), not in a job that may run
    alongside live engines.
    """
    offenders = [
        f"{path.name}:{number}: {code}"
        for path in _job_scripts(repo_root)
        for number, code in _code_lines(path.read_text())
        for pattern in INSTALL_COMMANDS
        if pattern.search(code)
    ]
    assert not offenders, (
        "job scripts must not install packages at run time:\n  " + "\n  ".join(offenders)
    )


def test_the_test_job_never_deactivates_its_venv(repo_root):
    """`deactivate` was the first half of the swap into the engine venv."""
    text = (repo_root / "slurm" / "tests" / "run_tests.sbatch").read_text()
    offenders = [
        f"line {number}: {code}"
        for number, code in _code_lines(text)
        if re.match(r"^deactivate\b", code)
    ]
    assert not offenders, "\n  ".join(offenders)


def test_the_vllm_tier_gates_on_the_engine_binary(repo_root):
    """A bare `-d` on the directory passes for an empty `uv venv .vllm`.

    That is what made the tier try to populate the venv rather than skip.
    """
    text = (repo_root / "slurm" / "tests" / "run_tests.sbatch").read_text()
    assert "-x \"${VLLM_VENV_PATH}/bin/vllm\"" in text
    assert "-d \"${VLLM_VENV_PATH}\"" not in text


def test_the_engine_venv_path_is_exported_for_the_gate(repo_root):
    """conftest reads VLLM_VENV_PATH, so the job must export it, not just set it."""
    text = (repo_root / "slurm" / "tests" / "run_tests.sbatch").read_text()
    assert re.search(r"^export VLLM_VENV_PATH$", text, re.M)


# NOTE: the ids are prefixed deliberately. `pytest_runtest_setup` in
# tests/conftest.py gates on `item.keywords`, which includes parametrize ids, so
# a bare id of "gpu"/"network"/"vllm" makes the case silently skip as though it
# needed that hardware.
@pytest.mark.parametrize(
    "tier",
    [
        pytest.param("slow", id="tier-slow"),
        pytest.param("network", id="tier-network"),
        pytest.param("gpu", id="tier-gpu"),
        pytest.param("vllm", id="tier-vllm"),
    ],
)
def test_every_tier_is_still_run(repo_root, tier):
    """Removing the install must not have removed the tier along with it."""
    text = (repo_root / "slurm" / "tests" / "run_tests.sbatch").read_text()
    assert re.search(rf"pytest .*-m [\"']?[^\"'\n]*\b{tier}\b", text), (
        f"the {tier} tier is no longer invoked"
    )
