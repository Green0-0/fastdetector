"""The transport-agnostic contract every offline-batch provider implements.

A batch run is three phases that must survive process death between them:
submit (expensive, already paid for), poll (up to 24h), fetch. The protocol is
split accordingly so :mod:`fastdetector.batch_state` can persist a job ID after
submit and a later process can resume at poll.
"""
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
    """Submit, poll, and fetch an offline batch of chat requests.

    Implementations must preserve the index mapping across the round trip:
    batch APIs return results in arbitrary order, so every implementation
    keys on its own request IDs and restores the caller's ordering itself.
    """

    name: str

    def submit(self, bodies: list[dict[str, Any]]) -> str:
        """Upload and start a batch.

        Args:
            bodies: Provider-specific request bodies, in caller order. The
                caller's index into this list is the identity that must be
                recoverable from fetch().

        Returns:
            An opaque job ID, durable enough to poll from another process.
        """
        ...

    def poll(self, job_id: str) -> tuple[bool, str]:
        """Check whether a batch has finished.

        Args:
            job_id: An ID previously returned by submit().

        Returns:
            Tuple of (is_terminal, status_string). A terminal batch may still
            have failed - that surfaces per-request in fetch().
        """
        ...

    def fetch(self, job_id: str, n_requests: int) -> list[BatchResult]:
        """Retrieve results for a terminal batch, restored to caller order.

        Args:
            job_id: An ID previously returned by submit().
            n_requests: Length of the original bodies list, so missing
                results can be padded rather than silently dropped.

        Returns:
            Exactly n_requests results, ordered by index.
        """
        ...


def order_results(
    found: dict[int, BatchResult], n_requests: int
) -> list[BatchResult]:
    """Restore caller ordering, padding any index the provider never returned.

    Batch APIs make no ordering guarantee and can drop requests entirely (an
    expired batch returns only what completed). Padding rather than truncating
    keeps the caller's row alignment intact.

    Args:
        found: Results keyed by their original index.
        n_requests: Expected result count.

    Returns:
        A dense list of length n_requests, ordered by index.
    """
    return [
        found.get(i, BatchResult(index=i, text="", prompt_tokens=0,
                                 completion_tokens=0, error="missing from batch output"))
        for i in range(n_requests)
    ]
