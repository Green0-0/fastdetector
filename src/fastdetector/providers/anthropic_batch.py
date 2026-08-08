"""Offline batch transport for the Anthropic Message Batches API.

Covers both the first-party API and Claude Platform on AWS. The latter is
Anthropic-operated access through AWS infrastructure - SigV4 auth, AWS
Marketplace billing - with same-day API parity, so the batch surface is
identical and only the client constructor differs. Model IDs stay bare
("claude-opus-5"); the "anthropic." prefix belongs to Amazon Bedrock, which is
a different product and does not expose the Message Batches API at all.

Batches are billed at roughly half the synchronous rate, with most completing
well inside the 24h window.
"""
import json
from typing import Any

from fastdetector.providers.base import BatchResult, order_results

# Per-batch ceiling. Larger inputs are split across several jobs.
MAX_REQUESTS_PER_JOB = 100_000


class AnthropicBatchProvider:
    """Submit/poll/fetch against the Anthropic Message Batches API."""

    def __init__(
        self,
        api_key: str | None = None,
        use_aws: bool = False,
        aws_region: str | None = None,
    ) -> None:
        """Build the appropriate client for the target transport.

        Args:
            api_key: Anthropic API key. Ignored when *use_aws* is set, since
                Claude Platform on AWS authenticates via the standard AWS
                credential chain.
            use_aws: Route through Claude Platform on AWS.
            aws_region: AWS region. Required for the AWS transport - unlike
                the Bedrock client there is no default fallback.

        Raises:
            ValueError: if the AWS transport is selected without a region.
        """
        if use_aws:
            from anthropic import AnthropicAWS

            if not aws_region:
                raise ValueError(
                    "aws_region is required for Claude Platform on AWS "
                    "(there is no default). Set it in the pipeline config or "
                    "export AWS_REGION."
                )
            self.name = "anthropic_aws"
            # workspace_id comes from ANTHROPIC_AWS_WORKSPACE_ID; credentials
            # resolve through the normal AWS chain.
            self.client = AnthropicAWS(aws_region=aws_region)
        else:
            from anthropic import Anthropic

            self.name = "anthropic"
            self.client = Anthropic(api_key=api_key)

    def submit(self, bodies: list[dict[str, Any]]) -> str:
        """Create one batch per chunk of requests.

        Args:
            bodies: Messages API request bodies in caller order.

        Returns:
            A JSON-encoded list of batch IDs. Opaque to the caller.
        """
        job_ids: list[str] = []
        for start in range(0, len(bodies), MAX_REQUESTS_PER_JOB):
            chunk = bodies[start:start + MAX_REQUESTS_PER_JOB]
            batch = self.client.messages.batches.create(
                requests=[
                    {"custom_id": _custom_id(start + offset), "params": body}
                    for offset, body in enumerate(chunk)
                ]
            )
            job_ids.append(batch.id)
            print(f"  Submitted batch {batch.id} ({len(chunk)} requests)", flush=True)

        return json.dumps(job_ids)

    def poll(self, job_id: str) -> tuple[bool, str]:
        """Report whether every chunk has finished processing.

        Args:
            job_id: Value previously returned by submit().

        Returns:
            Tuple of (all ended, summary of per-chunk statuses).
        """
        statuses = [
            self.client.messages.batches.retrieve(jid).processing_status
            for jid in json.loads(job_id)
        ]
        return all(s == "ended" for s in statuses), ", ".join(statuses)

    def fetch(self, job_id: str, n_requests: int) -> list[BatchResult]:
        """Stream every chunk's results and restore caller ordering.

        Args:
            job_id: Value previously returned by submit().
            n_requests: Expected result count.

        Returns:
            Exactly n_requests results, ordered by index.
        """
        found: dict[int, BatchResult] = {}

        for jid in json.loads(job_id):
            for entry in self.client.messages.batches.results(jid):
                result = _parse_entry(entry)
                found.setdefault(result.index, result)

        return order_results(found, n_requests)


def _custom_id(index: int) -> str:
    """Encode a caller index as a batch custom_id."""
    return f"req-{index}"


def _parse_entry(entry: Any) -> BatchResult:
    """Convert one streamed batch result into a BatchResult.

    Args:
        entry: An item yielded by messages.batches.results().

    Returns:
        The parsed result, with error set for anything but a usable success.
    """
    index = int(entry.custom_id.removeprefix("req-"))
    outcome = entry.result

    if outcome.type != "succeeded":
        # errored / canceled / expired all land here. Expired means the batch
        # ran out its window; those requests were never billed and can be
        # resubmitted.
        detail = getattr(getattr(outcome, "error", None), "type", None)
        return BatchResult(index, "", 0, 0, error=detail or outcome.type)

    message = outcome.message

    # Claude 5 ships elevated safety classifiers: a declined request is a
    # *successful* result carrying stop_reason "refusal" and empty content,
    # not an error. Without this branch it would silently become an empty
    # response with no explanation.
    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        return BatchResult(index, "", 0, 0, error=f"refusal ({category or 'unspecified'})")

    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    return BatchResult(
        index=index,
        text=text,
        prompt_tokens=message.usage.input_tokens,
        completion_tokens=message.usage.output_tokens,
    )
