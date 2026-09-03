import pytest


@pytest.mark.parametrize("name", ["gen.sbatch", "filter.sbatch"])
def test_long_online_jobs_checkpoint_before_the_time_limit(repo_root, name):
    text = (repo_root / "slurm" / name).read_text()
    assert text.count("#SBATCH --signal=B:USR1@300") == 1
    assert "#SBATCH --requeue" in text
    assert "source slurm/resume_job.sh" in text
    assert "fd_run_requeueable python" in text


def test_native_requeue_preserves_the_current_job(repo_root):
    text = (repo_root / "slurm" / "resume_job.sh").read_text()
    assert 'target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in text
    assert 'scontrol requeue "${target}"' in text
    assert "sbatch" not in text


def test_native_requeue_failure_is_not_hidden(repo_root):
    text = (repo_root / "slurm" / "resume_job.sh").read_text()
    assert 'if ! scontrol requeue "${target}"; then' in text
    assert 'echo "Failed to requeue ${target}." >&2' in text


def test_resubmission_has_a_persisted_attempt_cap(repo_root):
    text = (repo_root / "slurm" / "resume_job.sh").read_text()
    assert "SLURM_RESTART_COUNT:-0" in text
