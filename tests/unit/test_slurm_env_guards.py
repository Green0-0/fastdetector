"""Every ``set -u`` job script must guard the env vars that are often unset.

`slurm/tests/run_tests.sbatch` exited within seconds having run zero tests,
with only ``line 20: CPATH: unbound variable`` in the log. Under ``set -u``,
expanding an unset variable is fatal, and `CPATH` / `CPLUS_INCLUDE_PATH` are
unset on a stock login environment. The sibling scripts already used
``${VAR:-}``; this one was missed.

The issue suggested a CI shellcheck pass would catch the class. It does not:
shellcheck reports these files clean (exit 0 with SC1091 excluded) both before
and after the fix, because it cannot know which environment variables happen to
be set on the submitting host. So the guard is asserted directly here instead.

The check is textual on purpose -- it runs in the default tier without Slurm,
CUDA, or a cluster.
"""

import re

import pytest

#: Variables that are routinely unset on a login node, so expanding them bare
#: under ``set -u`` aborts the job. This is the append-to-a-search-path idiom:
#: `export X=/new/path:$X`.
OPTIONAL_ENV_VARS = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PKG_CONFIG_PATH",
    "PYTHONPATH",
)


def _job_scripts(repo_root):
    """Every committed sbatch script."""
    return sorted((repo_root / "slurm").rglob("*.sbatch"))


def _uses_nounset(text: str) -> bool:
    """True if the script enables ``set -u`` in any of its spellings."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("set "):
            continue
        flags = line.split("#", 1)[0].split()[1:]
        for flag in flags:
            if flag.startswith("-") and not flag.startswith("--") and "u" in flag:
                return True
    return False


def test_there_are_job_scripts_to_check(repo_root):
    """Guard against the glob silently matching nothing."""
    assert _job_scripts(repo_root)


def test_the_test_job_still_enables_nounset(repo_root):
    """The fix is the guard, not dropping ``set -u``.

    Removing ``-u`` would also make the job start, while giving up the
    protection that makes a typo'd variable loud instead of silent.
    """
    text = (repo_root / "slurm" / "tests" / "run_tests.sbatch").read_text()
    assert _uses_nounset(text)


@pytest.mark.parametrize("var", OPTIONAL_ENV_VARS)
def test_optional_env_vars_are_guarded_in_every_job_script(repo_root, var):
    """No ``set -u`` script may expand a frequently-unset variable bare.

    Matches ``$VAR`` and ``${VAR}`` but not ``${VAR:-...}`` or ``${VAR:=...}``,
    and not the assignment target ``export VAR=``.
    """
    bare = re.compile(r"\$" + var + r"\b|\$\{" + var + r"\}")
    offenders = []
    for path in _job_scripts(repo_root):
        text = path.read_text()
        if not _uses_nounset(text):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if bare.search(code):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert not offenders, (
        f"{var} is expanded without a default under `set -u`, which aborts the "
        f"job when it is unset:\n  " + "\n  ".join(offenders) + f"\nUse ${{{var}:-}} instead."
    )
