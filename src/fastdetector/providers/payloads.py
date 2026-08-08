"""Per-provider request-body construction.

The pipeline builds provider-neutral OpenAI-style message lists in
:func:`fastdetector.generator._build_messages`. These helpers translate one
such list into the body each provider's API expects.

Two invariants of the upstream message builder are relied on here, and
asserted rather than assumed:

- Messages alternate user/assistant, start with user, and end with user.
- No message carries the "system" role (the Prompt schema has no system field).

Both hold structurally for any turn count and any number of few-shot examples,
but Anthropic rejects a violation with a 400, so they are checked here where
the failure is cheap to diagnose.
"""
from typing import Any


def _validate(messages: list[dict[str, str]]) -> None:
    """Check the alternation and non-empty invariants Anthropic enforces.

    Args:
        messages: OpenAI-style message list.

    Raises:
        ValueError: if the list is empty, carries a system role, does not
            strictly alternate starting and ending on "user", or contains an
            empty content string.
    """
    if not messages:
        raise ValueError("Cannot build a request body from an empty message list.")

    for i, message in enumerate(messages):
        expected = "user" if i % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(
                f"Message {i} has role {message['role']!r}, expected {expected!r}. "
                f"Messages must alternate user/assistant starting with user."
            )
        if not message["content"].strip():
            raise ValueError(
                f"Message {i} ({message['role']}) has empty content. Anthropic "
                f"rejects empty text blocks; rows whose prior turn failed must "
                f"be dropped before this point, not replayed as empty assistants."
            )

    if messages[-1]["role"] != "user":
        raise ValueError("Message list must end with a user message.")


def to_openai(
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    model_name: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Build an OpenAI chat-completions request body.

    Args:
        messages: OpenAI-style message list.
        generation_params: Sampling/behaviour params already filtered to those
            this provider accepts.
        model_name: Model name, or the deployment name on Azure OpenAI.
        max_output_tokens: Optional output cap.

    Returns:
        A request body dict suitable for /v1/chat/completions.
    """
    _validate(messages)

    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        **generation_params,
    }
    if max_output_tokens is not None:
        body["max_completion_tokens"] = max_output_tokens
    return body


def to_anthropic(
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    model_name: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Build an Anthropic Messages request body.

    ``max_tokens`` is required by the API and bounds thinking *plus* response
    text together. Thinking is on by default on the Claude 5 family, so this
    value needs headroom beyond the expected answer length.

    Args:
        messages: OpenAI-style message list. Roles map 1:1 - no system message
            is ever produced by this pipeline, so none is hoisted.
        generation_params: Behaviour params already filtered to those this
            provider accepts. Sampling params are rejected by Claude 5 with a
            400 and must have been filtered out upstream.
        model_name: Model ID. Bare on both the first-party API and Claude
            Platform on AWS (no "anthropic." prefix - that is Bedrock).
        max_output_tokens: Required output cap.

    Returns:
        A request body dict suitable for the Messages API.

    Raises:
        ValueError: if max_output_tokens is unset, or a sampling param that
            Claude 5 rejects survived filtering.
    """
    _validate(messages)

    if max_output_tokens is None:
        raise ValueError(
            "max_output_tokens is required for Anthropic requests (the API "
            "rejects a Messages request without max_tokens)."
        )

    forbidden = {"temperature", "top_p", "top_k", "presence_penalty"} & generation_params.keys()
    if forbidden:
        raise ValueError(
            f"Sampling params {sorted(forbidden)} are removed in the Claude 5 "
            f"family and return a 400. They must be filtered out before this point."
        )

    return {
        "model": model_name,
        "max_tokens": max_output_tokens,
        "messages": messages,
        **generation_params,
    }


BUILDERS = {"openai": to_openai, "anthropic": to_anthropic}


def build_body(
    provider: str,
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    model_name: str,
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """Dispatch to the payload builder for *provider*.

    Args:
        provider: "openai" or "anthropic".
        messages: OpenAI-style message list.
        generation_params: Params already filtered for this provider.
        model_name: Model or deployment name.
        max_output_tokens: Output cap (required for Anthropic).

    Returns:
        A provider-specific request body dict.

    Raises:
        ValueError: if the provider is unknown.
    """
    if provider not in BUILDERS:
        raise ValueError(f"No payload builder for provider {provider!r}.")
    return BUILDERS[provider](messages, generation_params, model_name, max_output_tokens)
