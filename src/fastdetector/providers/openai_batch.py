import io
import json
from typing import Any

from fastdetector.providers.objects import BatchResult


class OpenAIBatchProvider:
    """Submit/poll/fetch against the OpenAI Batch API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Build the client for the target endpoint.

        Args:
            api_key: API key for the endpoint. None defers to OPENAI_API_KEY.
            base_url: Base URL of the chat-completions endpoint.
        """
        from openai import OpenAI

        self.name = "oai"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def submit(
        self,
        inputs: list[list[dict[str, str]]],
        generation_params: dict[str, Any],
        model_name: str,
        max_output_tokens: int | None,
    ) -> str:
        """Upload request chunks and start one batch job per chunk.

        Args:
            inputs: Conversations in caller order.
            generation_params: Params already filtered to this provider's set.
            model_name: Model name.
            max_output_tokens: Optional output cap.

        Returns:
            A JSON-encoded list of job IDs.
        """
        job_ids: list[str] = []
        for start in range(0, len(inputs), 50_000):
            chunk = inputs[start:start + 50_000]
            lines = []
            for offset, messages in enumerate(chunk):
                body: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    **generation_params,
                }
                if max_output_tokens is not None:
                    body["max_completion_tokens"] = max_output_tokens
                lines.append(json.dumps({
                    "custom_id": f"req-{start + offset}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }))
            payload = io.BytesIO(("\n".join(lines) + "\n").encode("utf-8"))
            payload.name = "batch_input.jsonl"

            uploaded = self.client.files.create(file=payload, purpose="batch")
            job = self.client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            job_ids.append(job.id)
            print(f"  Submitted batch {job.id} ({len(chunk)} requests)", flush=True)

        return json.dumps(job_ids)

    def poll(self, job_id: str) -> tuple[bool, str]:
        """Report whether every chunk of a batch has reached a terminal state.

        Args:
            job_id: Value previously returned by submit().

        Returns:
            Tuple of (all terminal, summary of per-chunk statuses).
        """
        statuses = [self.client.batches.retrieve(jid).status for jid in json.loads(job_id)]
        return (
            all(s in {"completed", "failed", "expired", "cancelled"} for s in statuses),
            ", ".join(statuses),
        )

    def fetch(self, job_id: str, n_requests: int) -> list[BatchResult]:
        """Download every chunk's output and restore caller ordering.

        Args:
            job_id: Value previously returned by submit().
            n_requests: Expected result count.

        Returns:
            Exactly n_requests results, ordered by index.
        """
        found: dict[int, BatchResult] = {}

        for jid in json.loads(job_id):
            job = self.client.batches.retrieve(jid)
            if job.status != "completed":
                print(
                    f"WARNING: batch {jid} ended as '{job.status}'; collecting whatever "
                    f"partial output exists.",
                    flush=True,
                )
            for file_id in (job.output_file_id, job.error_file_id):
                if not file_id:
                    continue
                for line in self.client.files.content(file_id).text.splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    index = int(record["custom_id"].removeprefix("req-"))
                    if index in found:
                        continue

                    response = record.get("response") or {}
                    body = response.get("body") or {}
                    choices = body.get("choices") or []

                    if record.get("error"):
                        found[index] = BatchResult(
                            index, "", 0, 0, error=str(record["error"]))
                    elif response.get("status_code") != 200:
                        found[index] = BatchResult(
                            index, "", 0, 0, error=f"HTTP {response.get('status_code')}")
                    elif not choices:
                        found[index] = BatchResult(
                            index, "", 0, 0,
                            error="no choices (possibly content-filtered)")
                    else:
                        usage = body.get("usage") or {}
                        found[index] = BatchResult(
                            index=index,
                            text=(choices[0].get("message") or {}).get("content") or "",
                            prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0),
                        )

        return [
            found.get(i, BatchResult(i, "", 0, 0, error="missing from batch output"))
            for i in range(n_requests)
        ]
