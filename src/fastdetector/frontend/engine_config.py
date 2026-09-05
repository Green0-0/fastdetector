from enum import Enum


class EngineConfig(str, Enum):
    """Specifies an engine with its supported properties."""

    VLLM = "vllm"
    APHRODITE = "aphrodite"
    OAI = "oai"
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

        Returns:
            True if the engine is a hosted API, False otherwise.
        """
        return not self.is_local_server

    @property
    def provider(self) -> str | None:
        """Which request-payload dialect this engine speaks.

        Returns:
            "openai", "anthropic", or None for local-server engines.
        """
        if self == EngineConfig.OAI:
            return "openai"
        if self in (EngineConfig.ANTHROPIC, EngineConfig.ANTHROPIC_AWS):
            return "anthropic"
        return None

    @property
    def valid_sampling_params(self) -> list[str]:
        """Returns all valid sampling parameters for this engine.

        Returns:
            List of supported parameter names.
        """
        base_params = [
            "temperature", "top_p", "top_k", "presence_penalty",
            "repetition_penalty", "disable_thinking",
        ]

        if self == EngineConfig.VLLM:
            return base_params
        if self == EngineConfig.APHRODITE:
            return base_params + ["top_a", "xtc_probability", "nsigma"]
        return ["disable_thinking"]
