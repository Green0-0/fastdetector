"""Small append-only checkpoints for online generation."""

import fcntl
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class CheckpointRecord:
    row_id: int
    text: str
    prompt_tokens: int
    completion_tokens: int


class TurnLog:
    """Saved successful responses for one turn."""

    def __init__(
        self, path: Path, turn_index: int, flush_records: int, flush_seconds: float
    ) -> None:
        self.path = path
        self.turn_index = turn_index
        self.records = self._load()
        self._handle: TextIO = path.open("a", encoding="utf-8")
        self._flush_records = flush_records
        self._flush_seconds = flush_seconds
        self._pending = 0
        self._last_flush = time.monotonic()

    def _load(self) -> dict[int, CheckpointRecord]:
        if not self.path.exists():
            return {}

        records = {}
        size = self.path.stat().st_size
        valid_bytes = 0
        with self.path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if handle.tell() == size and not line.endswith(b"\n"):
                    print(f"WARNING: Ignoring torn final line in {self.path}.")
                    break
                try:
                    value = json.loads(line)
                    record = CheckpointRecord(**value)
                    if not record.text.strip():
                        raise ValueError("empty response")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid checkpoint record in {self.path}:{line_number}: {exc}"
                    ) from exc
                records[record.row_id] = record
                valid_bytes += len(line)

        if valid_bytes < size:
            with self.path.open("r+b") as handle:
                handle.truncate(valid_bytes)
        return records

    def get(self, row_id: int) -> CheckpointRecord | None:
        return self.records.get(row_id)

    def pending_count(self, row_ids: list[int]) -> int:
        return sum(row_id not in self.records for row_id in row_ids)

    def append(
        self, row_id: int, text: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        if not text.strip():
            return
        record = CheckpointRecord(row_id, text, prompt_tokens, completion_tokens)
        self._handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        # Keep completed responses out of Python's buffer if the process is
        # terminated; fsync remains batched to avoid one disk sync per request.
        self._handle.flush()
        self.records[row_id] = record
        self._pending += 1
        if (
            self._pending >= self._flush_records
            or time.monotonic() - self._last_flush >= self._flush_seconds
        ):
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()


class GenerationCheckpoint:
    """Fingerprint plus one append-only log per turn."""

    def __init__(
        self,
        base_dir: str,
        run_key: str,
        *,
        flush_records: int = 50,
        flush_seconds: float = 30.0,
    ) -> None:
        self.path = Path(base_dir) / run_key
        self._flush_records = flush_records
        self._flush_seconds = flush_seconds
        self._turns: dict[int, TurnLog] = {}
        self._lock_handle: TextIO | None = None
        self._prepared = False

    def prepare(self, inputs: dict[str, Any]) -> str:
        """Start fresh when anything affecting generation changed."""
        if self._lock_handle is not None:
            raise RuntimeError("checkpoint is already prepared")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.parent / f".{self.path.name}.lock"
        self._lock_handle = lock_path.open("a", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(f"Checkpoint is already in use: {self.path}") from exc

        fingerprint = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        meta_path = self.path / "meta.json"

        if self.path.exists():
            try:
                matches = json.loads(meta_path.read_text())["fingerprint"] == fingerprint
            except (OSError, KeyError, json.JSONDecodeError):
                matches = False
            if not matches:
                archived = self.path.with_name(f"{self.path.name}.stale-{time.time_ns()}")
                self.path.rename(archived)
                print(f"Inputs changed; archived old checkpoint at {archived}.")

        self.path.mkdir(parents=True, exist_ok=True)
        if not meta_path.exists():
            temporary = meta_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"fingerprint": fingerprint, "inputs": inputs}))
            os.replace(temporary, meta_path)
        self._prepared = True
        return fingerprint

    def turn(self, turn_index: int) -> TurnLog:
        if not self._prepared:
            raise RuntimeError("checkpoint must be prepared before use")
        if turn_index not in self._turns:
            self._turns[turn_index] = TurnLog(
                self.path / f"turn_{turn_index}.jsonl",
                turn_index,
                self._flush_records,
                self._flush_seconds,
            )
        return self._turns[turn_index]

    def flush(self) -> None:
        for turn in self._turns.values():
            turn.flush()

    def _close_turns(self) -> None:
        for turn in self._turns.values():
            turn.close()
        self._turns.clear()

    def close(self) -> None:
        self._close_turns()
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle, fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
        self._prepared = False

    def retire(self) -> None:
        self._close_turns()
        try:
            if self.path.exists():
                shutil.rmtree(self.path)
        finally:
            self.close()
        print(f"Retired generation checkpoint {self.path}.")
