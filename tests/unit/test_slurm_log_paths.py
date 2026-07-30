import re

import pytest

#: `%x_%j` for a plain job, `%x_%A_%a` for an array job.
LOG_PATTERN = re.compile(r"^slurm/logs/%x_(%j|%A_%a)\.(out|err)$")

DIRECTIVE = re.compile(r"^#SBATCH\s+--(output|error)=(\S+)")


def _job_scripts(repo_root):
    return sorted((repo_root / "slurm").rglob("*.sbatch"))


def _log_directives(text):
    """Yield ``(kind, path)`` for each --output/--error directive."""
    for line in text.splitlines():
        match = DIRECTIVE.match(line.strip())
        if match:
            yield match.group(1), match.group(2)


def test_there_are_job_scripts_to_check(repo_root):
    assert _job_scripts(repo_root)


@pytest.mark.parametrize("kind", ["output", "error"])
def test_every_job_script_declares_a_log_path(repo_root, kind):
    """A script with no directive falls back to the CWD, which is the bug."""
    missing = [
        path.name
        for path in _job_scripts(repo_root)
        if kind not in {k for k, _ in _log_directives(path.read_text())}
    ]
    assert not missing, f"no --{kind} directive in: {missing}"


def test_all_logs_land_in_slurm_logs(repo_root):
    """Otherwise job output lands wherever sbatch was invoked from."""
    offenders = []
    for path in _job_scripts(repo_root):
        for kind, value in _log_directives(path.read_text()):
            if not value.startswith("slurm/logs/"):
                offenders.append(f"{path.name}: --{kind}={value}")

    assert not offenders, (
        "job output must go to slurm/logs/ so it does not accumulate in the "
        "working directory:\n  " + "\n  ".join(offenders)
    )


def test_log_names_follow_the_house_pattern(repo_root):
    """``%x`` keeps the filename tied to --job-name instead of a hardcoded string."""
    offenders = []
    for path in _job_scripts(repo_root):
        for kind, value in _log_directives(path.read_text()):
            if not LOG_PATTERN.match(value):
                offenders.append(f"{path.name}: --{kind}={value}")

    assert not offenders, (
        "expected slurm/logs/%x_%j.<ext> (or %x_%A_%a for array jobs):\n  "
        + "\n  ".join(offenders)
    )


def test_array_jobs_use_the_array_aware_pattern(repo_root):
    """``%j`` in an array job overwrites: every task shares one job id.

    Array tasks need ``%A_%a`` to get one log per task.
    """
    offenders = []
    for path in _job_scripts(repo_root):
        text = path.read_text()
        is_array = any(
            line.strip().startswith("#SBATCH") and "--array" in line
            for line in text.splitlines()
        )
        if not is_array:
            continue
        for kind, value in _log_directives(text):
            if "%A_%a" not in value:
                offenders.append(f"{path.name}: --{kind}={value}")

    assert not offenders, (
        "array jobs must use %A_%a or tasks overwrite each other's logs:\n  "
        + "\n  ".join(offenders)
    )


def test_the_log_directory_is_committed(repo_root):
    """`sbatch` fails outright if the log directory does not exist."""
    assert (repo_root / "slurm" / "logs").is_dir()
    assert (repo_root / "slurm" / "logs" / ".gitkeep").is_file()
