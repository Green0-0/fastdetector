import json
from typing import Any

from fastdetector.providers.objects import BatchResult


class AnthropicBatchProvider:
    """Submit/poll/fetch against the Anthropic Message Batches API."""

    def __init__(
        self,
        api_key: str | None = None,
        use_aws: bool = False,
        aws_region: str | None = None,
    ) -> None:
        """Build the client for the target transport.

        Args:
            api_key: Anthropic API key. Ignored when *use_aws* is set.
            use_aws: Route through Claude Platform on AWS.
            aws_region: AWS region, required for the AWS transport.

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
            self.client = AnthropicAWS(aws_region=aws_region)
        else:
            from anthropic import Anthropic

            self.name = "anthropic"
            self.client = Anthropic(api_key=api_key)

    def submit(
        self,
        inputs: list[list[dict[str, str]]],
        generation_params: dict[str, Any],
        model_name: str,
        max_output_tokens: int | None,
    ) -> str:
        """Create one batch per chunk of requests.

        Args:
            inputs: Conversations in caller order.
            generation_params: Params already filtered to this provider's set.
            model_name: Model ID, bare (the "anthropic." prefix is Bedrock's).
            max_output_tokens: Required output cap; bounds thinking plus
                response text together.

        Returns:
            A JSON-encoded list of batch IDs.

        Raises:
            ValueError: if max_output_tokens is unset, or a sampling param that
                the Claude 5 family rejects survived filtering.
        """
        if max_output_tokens is None:
            raise ValueError(
                "max_output_tokens is required for Anthropic requests (the API "
                "rejects a Messages request without max_tokens)."
            )
        forbidden = {
            "temperature", "top_p", "top_k", "presence_penalty"
        } & generation_params.keys()
        if forbidden:
            raise ValueError(
                f"Sampling params {sorted(forbidden)} are removed in the Claude 5 "
                f"family and return a 400."
            )

        job_ids: list[str] = []
        for start in range(0, len(inputs), 100_000):
            chunk = inputs[start:start + 100_000]
            batch = self.client.messages.batches.create(
                requests=[
                    {
                        "custom_id": f"req-{start + offset}",
                        "params": {
                            "model": model_name,
                            "max_tokens": max_output_tokens,
                            "messages": messages,
                            **generation_params,
                        },
                    }
                    for offset, messages in enumerate(chunk)
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
                index = int(entry.custom_id.removeprefix("req-"))
                if index in found:
                    continue
                outcome = entry.result

                if outcome.type != "succeeded":
                    inner = getattr(getattr(outcome, "error", None), "error", None)
                    found[index] = BatchResult(
                        index, "", 0, 0,
                        error=getattr(inner, "type", None) or outcome.type)
                    continue

                message = outcome.message

                if message.stop_reason == "refusal":
                    category = getattr(
                        getattr(message, "stop_details", None), "category", None)
                    found[index] = BatchResult(
                        index, "", 0, 0,
                        error=f"refusal ({category or 'unspecified'})")
                    continue

                found[index] = BatchResult(
                    index=index,
                    text="".join(
                        block.text for block in message.content
                        if block.type == "text"
                    ),
                    prompt_tokens=message.usage.input_tokens,
                    completion_tokens=message.usage.output_tokens,
                )

        return [
            found.get(i, BatchResult(i, "", 0, 0, error="missing from batch output"))
            for i in range(n_requests)
        ]
