"""Durable record of submitted batch jobs, so a killed run can resume.

Once a batch is submitted it has been paid for, and it may take up to 24 hours
to complete. If the driving process dies in that window the job ID is the only
way back to the work - losing it means paying twice. Every write is therefore
atomic (temp file + rename) so a process killed mid-write leaves the previous
state intact rather than a truncated file.

State is keyed by (shard, turn) because build_dataset issues one batch per chat
turn and turns are strictly sequential.
"""
import json
import os
import tempfile
from typing import Any


class BatchState:
    """Read/write access to one shard's batch job records."""

    def __init__(self, state_dir: str, run_key: str) -> None:
        """Open (or create) the state file for a run.

        Args:
            state_dir: Directory holding state files. Created if absent.
            run_key: Identifier unique to this (engine, model, shard)
                combination, used as the file name stem.
        """
        self.path = os.path.join(state_dir, f"{_slug(run_key)}.json")
        os.makedirs(state_dir, exist_ok=True)
        self._data: dict[str, Any] = {}
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as handle:
                    self._data = json.load(handle)
            except (json.JSONDecodeError, OSError) as e:
                # A corrupt state file must not silently become "no prior job",
                # which would resubmit and double-bill.
                raise RuntimeError(
                    f"Batch state file {self.path} exists but could not be read: {e}. "
                    f"Inspect it (or delete it, accepting that in-flight jobs are "
                    f"orphaned and will be resubmitted) before rerunning."
                ) from e

    def get(self, turn: int) -> dict[str, Any] | None:
        """Return the recorded job for *turn*, if one was submitted.

        Args:
            turn: Chat turn index.

        Returns:
            The stored record, or None if this turn has not been submitted.
        """
        return self._data.get(_turn_key(turn))

    def record(self, turn: int, job_id: str, provider: str, n_requests: int) -> None:
        """Persist a freshly submitted job before doing anything else.

        Args:
            turn: Chat turn index.
            job_id: Provider job ID.
            provider: Provider name, recorded so a resume can detect a config
                change that invalidates the stored ID.
            n_requests: Number of requests in the batch, needed to pad results.
        """
        self._data[_turn_key(turn)] = {
            "job_id": job_id,
            "provider": provider,
            "n_requests": n_requests,
            "complete": False,
        }
        self._flush()

    def mark_complete(self, turn: int) -> None:
        """Mark a turn's results as collected.

        Args:
            turn: Chat turn index.
        """
        key = _turn_key(turn)
        if key in self._data:
            self._data[key]["complete"] = True
            self._flush()

    def _flush(self) -> None:
        """Write state atomically so a crash cannot truncate the file."""
        directory = os.path.dirname(self.path) or "."
        handle = tempfile.NamedTemporaryFile(
            "w", dir=directory, delete=False, encoding="utf-8", suffix=".tmp"
        )
        try:
            with handle:
                json.dump(self._data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except BaseException:
            if os.path.exists(handle.name):
                os.unlink(handle.name)
            raise


def _turn_key(turn: int) -> str:
    """Return the state-dict key for a chat turn."""
    return f"turn_{turn}"


def _slug(text: str) -> str:
    """Reduce an identifier to a filesystem-safe stem."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)
