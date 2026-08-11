import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class BatchResult:
    """One request's outcome, keyed back to its position in the input list.

    Args:
        index: Position in the list originally handed to submit().
        text: Response text, or "" when the request did not succeed.
        prompt_tokens: Input tokens billed.
        completion_tokens: Output tokens billed.
        error: Short reason string when the request failed, else None.
    """

    index: int
    text: str
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True if this request produced no usable text."""
        return self.error is not None or not self.text


class BatchProvider(Protocol):
    """Submit, poll, and fetch an offline batch of chat requests."""

    name: str

    def submit(
        self,
        inputs: list[list[dict[str, str]]],
        generation_params: dict[str, Any],
        model_name: str,
        max_output_tokens: int | None,
    ) -> str:
        """Build this provider's request bodies and start a batch.

        Args:
            inputs: Conversations in caller order.
            generation_params: Params already filtered to this provider's set.
            model_name: Model or deployment name.
            max_output_tokens: Output cap; required by some providers.

        Returns:
            An opaque job ID, durable enough to poll from another process.
        """
        ...

    def poll(self, job_id: str) -> tuple[bool, str]:
        """Check whether a batch has finished.

        Args:
            job_id: An ID previously returned by submit().

        Returns:
            Tuple of (is_terminal, status_string).
        """
        ...

    def fetch(self, job_id: str, n_requests: int) -> list[BatchResult]:
        """Retrieve results for a terminal batch, restored to caller order.

        Args:
            job_id: An ID previously returned by submit().
            n_requests: Length of the original bodies list.

        Returns:
            Exactly n_requests results, ordered by index.
        """
        ...


class BatchState:
    """Read/write access to one run's batch job records, keyed by chat turn."""

    def __init__(self, state_dir: str, run_key: str) -> None:
        """Open (or create) the state file for a single run.

        Args:
            state_dir: Directory holding state files. Created if absent.
            run_key: Identifier distinguishing this run's file from those of
                other runs sharing the directory.

        Raises:
            RuntimeError: if the state file exists but cannot be read.
        """
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_key)
        self.path = os.path.join(state_dir, f"{stem}.json")
        os.makedirs(state_dir, exist_ok=True)
        self._data: dict[str, Any] = {}
        if os.path.isfile(self.path):
            try:
                with open(self.path, encoding="utf-8") as handle:
                    self._data = json.load(handle)
            except (json.JSONDecodeError, OSError) as e:
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
        return self._data.get(f"turn_{turn}")

    def record(self, turn: int, job_id: str, provider: str, n_requests: int) -> None:
        """Persist a freshly submitted job before doing anything else.

        Args:
            turn: Chat turn index.
            job_id: Provider job ID.
            provider: Provider name.
            n_requests: Number of requests in the batch.
        """
        self._data[f"turn_{turn}"] = {
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
        key = f"turn_{turn}"
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
