from enum import Enum


class EngineConfig(str, Enum):
    """Specifies an engine with its supported properties."""

    VLLM = "vllm"
    APHRODITE = "aphrodite"
    OAI = "oai"

    @property
    def is_local_server(self) -> bool:
        """True if this engine requires launching a local vLLM/Aphrodite server.

        Returns:
            True if local server launch is required, False otherwise.
        """
        return self in (EngineConfig.VLLM, EngineConfig.APHRODITE)

    @property
    def is_proprietary(self) -> bool:
        """True if this engine is a proprietary API model (not a local server).

        Proprietary models use different sampling-param semantics: they don't accept
        temperature/top_p/top_k/presence_penalty, and disable_thinking is
        translated to reasoning_effort="none" instead of a chat_template_kwarg.

        Returns:
            True if the engine is a proprietary API (e.g. OAI), False otherwise.
        """
        return self == EngineConfig.OAI

    @property
    def valid_sampling_params(self) -> list[str]:
        """Returns all valid sampling parameters for this engine.

        Returns:
            List of supported parameter names.
        """
        base_params = ["temperature", "top_p", "top_k", "presence_penalty", "disable_thinking"]

        if self == EngineConfig.VLLM:
            return base_params
        elif self == EngineConfig.APHRODITE:
            return base_params + ["top_a", "xtc_probability", "nsigma"]
        elif self == EngineConfig.OAI:
            return ["disable_thinking"]
        return []