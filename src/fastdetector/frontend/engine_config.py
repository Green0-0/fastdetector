from enum import Enum


class EngineConfig(str, Enum):
    """Specifies an engine with its supported properties.

    An engine names the *vendor and transport* (where requests are sent and
    with which SDK). Whether those requests go through the synchronous or the
    offline-batch lifecycle is a separate axis, set by ``PipeConfig.batch``.
    """

    VLLM = "vllm"
    APHRODITE = "aphrodite"

    # OpenAI-family. Both speak the same Files/Batches surface via the
    # `openai` SDK; only the client constructor and the meaning of
    # `model_name` differ (Azure takes a deployment name).
    OAI = "oai"
    AZURE_OAI = "azure_oai"

    # Anthropic-family. Both speak the same Message Batches surface via the
    # `anthropic` SDK; ANTHROPIC_AWS is Claude Platform on AWS, which is
    # Anthropic-operated with AWS Marketplace billing and SigV4 auth.
    ANTHROPIC = "anthropic"
    ANTHROPIC_AWS = "anthropic_aws"

    @property
    def is_local_server(self) -> bool:
        """True if this engine requires launching a local vLLM/Aphrodite server.

        Returns:
            True if local server launch is required, False otherwise.
        """
        return self in (EngineConfig.VLLM, EngineConfig.APHRODITE)

    @property
    def is_proprietary(self) -> bool:
        """True if this engine is a hosted API rather than a local server.

        Proprietary engines skip tokenizer-based length limits (there is no
        local tokenizer to load) and use per-provider sampling-param rules.

        Returns:
            True if the engine is a hosted API, False otherwise.
        """
        return not self.is_local_server

    @property
    def provider(self) -> str | None:
        """Which request-payload dialect this engine speaks.

        Transport (native vs Azure vs Claude Platform on AWS) varies
        independently; the payload shape is a property of the provider alone,
        which is why sampling-param validity keys off this rather than off the
        engine member.

        Returns:
            "openai", "anthropic", or None for local-server engines.
        """
        if self in (EngineConfig.OAI, EngineConfig.AZURE_OAI):
            return "openai"
        if self in (EngineConfig.ANTHROPIC, EngineConfig.ANTHROPIC_AWS):
            return "anthropic"
        return None

    @property
    def valid_sampling_params(self) -> list[str]:
        """Returns all valid sampling parameters for this engine.

        Anthropic models reject temperature/top_p/top_k outright (they were
        removed in the Claude 5 family and return a 400), so the list is not
        merely a preference - sending anything outside it fails the request.

        Returns:
            List of supported parameter names.
        """
        base_params = ["temperature", "top_p", "top_k", "presence_penalty", "disable_thinking"]

        if self == EngineConfig.VLLM:
            return base_params
        if self == EngineConfig.APHRODITE:
            return base_params + ["top_a", "xtc_probability", "nsigma"]
        if self.provider in ("openai", "anthropic"):
            return ["disable_thinking"]
        return []
