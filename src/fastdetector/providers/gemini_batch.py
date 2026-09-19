import json
import os
import tempfile
from typing import Any

from fastdetector.providers.objects import BatchResult


# Google accepts input files up to 2 GB. Leave headroom for limits being
# interpreted differently by intermediate upload infrastructure.
_MAX_INPUT_FILE_BYTES = 1_900_000_000
_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def _state_name(job: Any) -> str:
    """Return an SDK batch job's state as its stable API string."""
    state = getattr(job, "state", None)
    return getattr(state, "name", None) or str(state)


def _error_text(error: Any) -> str:
    """Turn either an SDK error object or JSON status object into a short reason."""
    if error is None:
        return "unknown error"
    if isinstance(error, dict):
        return str(
            error.get("message")
            or error.get("status")
            or error.get("code")
            or error
        )
    return str(getattr(error, "message", None) or error)


def _gemini_request(
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Convert an OpenAI-style conversation to GenerateContent JSON."""
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []

    for message in messages:
        role = message["role"]
        part = {"text": message["content"]}
        if role in {"system", "developer"}:
            system_parts.append(part)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [part]})
        elif role == "user":
            contents.append({"role": "user", "parts": [part]})
        else:
            raise ValueError(f"Unsupported Gemini message role: {role!r}")

    request: dict[str, Any] = {"contents": contents}
    if system_parts:
        request["system_instruction"] = {"parts": system_parts}

    generation_config = dict(generation_params)
    if max_output_tokens is not None:
        generation_config["max_output_tokens"] = max_output_tokens
    if generation_config:
        request["generation_config"] = generation_config
    return request


def _result_from_record(record: dict[str, Any]) -> BatchResult:
    """Parse one keyed Gemini batch-output JSON object."""
    key = record.get("key")
    if key is None:
        key = (record.get("metadata") or {}).get("key")
    if not isinstance(key, str) or not key.startswith("req-"):
        raise ValueError(f"Gemini batch result has no valid request key: {key!r}")
    index = int(key.removeprefix("req-"))

    record_error = record.get("error") or record.get("status")
    if record_error is not None:
        return BatchResult(index, "", 0, 0, error=_error_text(record_error))

    response = record.get("response")
    # Accept an unwrapped GenerateContentResponse as a defensive fallback.
    if response is None and (
        "candidates" in record or "promptFeedback" in record or "prompt_feedback" in record
    ):
        response = record
    if not isinstance(response, dict):
        return BatchResult(index, "", 0, 0, error="batch result contained no response")

    candidates = response.get("candidates") or []
    prompt_feedback = response.get("promptFeedback") or response.get("prompt_feedback") or {}
    if not candidates:
        block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
        reason = f"prompt blocked ({block_reason})" if block_reason else "response contained no candidates"
        return BatchResult(index, "", 0, 0, error=reason)

    candidate = candidates[0]
    content = candidate.get("content") or {}
    text = "".join(
        str(part.get("text") or "")
        for part in content.get("parts") or []
        if not part.get("thought")
    )
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    prompt_tokens = usage.get("promptTokenCount", usage.get("prompt_token_count", 0)) or 0
    visible_tokens = usage.get(
        "candidatesTokenCount", usage.get("candidates_token_count", 0)
    ) or 0
    thought_tokens = usage.get("thoughtsTokenCount", usage.get("thoughts_token_count", 0)) or 0

    if not text.strip():
        finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
        reason = "response contained no usable text"
        if finish_reason:
            reason += f" (finish reason: {finish_reason})"
        return BatchResult(
            index, "", int(prompt_tokens), int(visible_tokens + thought_tokens), error=reason
        )

    return BatchResult(
        index=index,
        text=text,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(visible_tokens + thought_tokens),
    )


class GeminiBatchProvider:
    """Submit/poll/fetch against the Gemini Developer API Batch API."""

    def __init__(self, api_key: str | None = None) -> None:
        """Build a Gemini Developer API client.

        Args:
            api_key: Gemini API key. None lets the SDK inspect its standard
                GOOGLE_API_KEY and GEMINI_API_KEY environment variables.
        """
        from google import genai

        self.name = "gemini"
        self.client = genai.Client(api_key=api_key)

    def submit(
        self,
        inputs: list[list[dict[str, str]]],
        generation_params: dict[str, Any],
        model_name: str,
        max_output_tokens: int | None,
    ) -> str:
        """Upload keyed JSONL request files and start Gemini batch jobs."""
        if not inputs:
            raise ValueError("Gemini batch submission requires at least one request")

        job_ids: list[str] = []
        temp_path: str | None = None
        handle = None
        chunk_start = 0
        chunk_count = 0
        chunk_bytes = 0

        def open_chunk() -> None:
            nonlocal handle, temp_path, chunk_count, chunk_bytes
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", suffix=".jsonl", delete=False
            )
            temp_path = handle.name
            chunk_count = 0
            chunk_bytes = 0

        def submit_chunk() -> None:
            nonlocal handle, temp_path, chunk_start
            if handle is None or temp_path is None or chunk_count == 0:
                return
            handle.close()
            try:
                uploaded = self.client.files.upload(
                    file=temp_path,
                    config={
                        "display_name": f"fastdetector-requests-{chunk_start}",
                        "mime_type": "jsonl",
                    },
                )
                job = self.client.batches.create(
                    model=model_name,
                    src=uploaded.name,
                    config={"display_name": f"fastdetector-batch-{chunk_start}"},
                )
                job_ids.append(job.name)
                print(
                    f"  Submitted batch {job.name} ({chunk_count} requests)",
                    flush=True,
                )
            finally:
                os.unlink(temp_path)
                handle = None
                temp_path = None
            chunk_start += chunk_count

        try:
            open_chunk()
            for index, messages in enumerate(inputs):
                record = {
                    "key": f"req-{index}",
                    "request": _gemini_request(
                        messages, generation_params, max_output_tokens
                    ),
                }
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                line_bytes = len(line.encode("utf-8"))
                if line_bytes > _MAX_INPUT_FILE_BYTES:
                    raise ValueError(
                        f"Gemini request req-{index} exceeds the 2 GB batch-file limit"
                    )
                if chunk_count and chunk_bytes + line_bytes > _MAX_INPUT_FILE_BYTES:
                    submit_chunk()
                    open_chunk()
                assert handle is not None
                handle.write(line)
                chunk_count += 1
                chunk_bytes += line_bytes
            submit_chunk()
        finally:
            if handle is not None:
                handle.close()
            if temp_path is not None and os.path.exists(temp_path):
                os.unlink(temp_path)

        return json.dumps(job_ids)

    def poll(self, job_id: str) -> tuple[bool, str]:
        """Report whether every Gemini batch chunk is terminal."""
        statuses = [
            _state_name(self.client.batches.get(name=name))
            for name in json.loads(job_id)
        ]
        return all(status in _TERMINAL_STATES for status in statuses), ", ".join(statuses)

    def fetch(self, job_id: str, n_requests: int) -> list[BatchResult]:
        """Download result JSONL files and restore original caller ordering."""
        found: dict[int, BatchResult] = {}

        for name in json.loads(job_id):
            job = self.client.batches.get(name=name)
            state = _state_name(job)
            if state != "JOB_STATE_SUCCEEDED":
                detail = _error_text(getattr(job, "error", None))
                print(
                    f"WARNING: batch {name} ended as '{state}' ({detail}); collecting "
                    "whatever partial output exists.",
                    flush=True,
                )

            destination = getattr(job, "dest", None)
            file_name = getattr(destination, "file_name", None)
            if not file_name:
                continue
            payload = self.client.files.download(file=file_name)
            if payload is None:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            for line in payload.splitlines():
                if not line.strip():
                    continue
                result = _result_from_record(json.loads(line))
                found.setdefault(result.index, result)

        return [
            found.get(i, BatchResult(i, "", 0, 0, error="missing from batch output"))
            for i in range(n_requests)
        ]
