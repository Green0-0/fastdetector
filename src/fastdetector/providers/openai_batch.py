"""Offline batch transport for the OpenAI Batch API.

Covers both the first-party API and Azure OpenAI Global Batch: they expose the
same Files/Batches surface through the same `openai` SDK, so only the client
constructor differs and `model_name` means a deployment name on Azure.

Both are billed at roughly half the synchronous rate with a 24h completion
window.
"""
import io
import json
from typing import Any

from fastdetector.providers.base import BatchResult, order_results

# Per-file ceilings on the Batch API. Larger inputs are split across several
# jobs rather than rejected, since a shard's size is decided by the sharding
# step and is not the batch layer's business to constrain.
MAX_REQUESTS_PER_JOB = 50_000

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchProvider:
    """Submit/poll/fetch against the OpenAI (or Azure OpenAI) Batch API."""

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Build the appropriate client for the target transport.

        Args:
            api_key: API key for the endpoint.
            azure_endpoint: Azure resource endpoint. When set, the Azure client
                is used and *model_name* must be a deployment name.
            azure_api_version: Azure API version, required alongside the endpoint.
            base_url: Override for the first-party base URL.

        Raises:
            ValueError: if only one half of the Azure configuration is given.
        """
        if bool(azure_endpoint) != bool(azure_api_version):
            raise ValueError(
                "azure_endpoint and azure_api_version must be set together."
            )

        if azure_endpoint:
            from openai import AzureOpenAI

            self.name = "azure_oai"
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=azure_api_version,
            )
        else:
            from openai import OpenAI

            self.name = "oai"
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def submit(self, bodies: list[dict[str, Any]]) -> str:
        """Upload request chunks and start one batch job per chunk.

        Args:
            bodies: Chat-completions request bodies in caller order.

        Returns:
            A JSON-encoded list of job IDs. Opaque to the caller; only this
            provider interprets it.
        """
        job_ids: list[str] = []
        for start in range(0, len(bodies), MAX_REQUESTS_PER_JOB):
            chunk = bodies[start:start + MAX_REQUESTS_PER_JOB]
            lines = [
                json.dumps({
                    "custom_id": _custom_id(start + offset),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                })
                for offset, body in enumerate(chunk)
            ]
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
        return all(s in TERMINAL_STATUSES for s in statuses), ", ".join(statuses)

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
                if file_id:
                    for line in self.client.files.content(file_id).text.splitlines():
                        if line.strip():
                            result = _parse_line(line)
                            found.setdefault(result.index, result)

        return order_results(found, n_requests)


def _custom_id(index: int) -> str:
    """Encode a caller index as a batch custom_id."""
    return f"req-{index}"


def _parse_line(line: str) -> BatchResult:
    """Convert one output JSONL line into a BatchResult.

    Args:
        line: A single line from a batch output or error file.

    Returns:
        The parsed result, with error set when the request did not succeed.
    """
    record = json.loads(line)
    index = int(record["custom_id"].removeprefix("req-"))

    if record.get("error"):
        return BatchResult(index, "", 0, 0, error=str(record["error"]))

    response = record.get("response") or {}
    if response.get("status_code") != 200:
        return BatchResult(index, "", 0, 0, error=f"HTTP {response.get('status_code')}")

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return BatchResult(index, "", 0, 0, error="no choices (possibly content-filtered)")

    usage = body.get("usage") or {}
    return BatchResult(
        index=index,
        text=(choices[0].get("message") or {}).get("content") or "",
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )
