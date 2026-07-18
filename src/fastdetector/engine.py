"""Engine type enumeration for LLM inference backends.

Centralizes the string-based engine selection logic that was previously
scattered across pipe.py, llm_utils.py, and generator.py as ad-hoc
string equality checks.

Engine is a str enum so it serializes cleanly to TOML/JSON (the value is
the lowercase string name, e.g. "vllm", "aphrodite", "oai").
"""

from enum import Enum


class Engine(str, Enum):
    """Supported LLM engine backends.

    Values are lowercase strings so they match TOML config values directly.
    """

    VLLM = "vllm"
    APHRODITE = "aphrodite"
    OAI = "oai"

    @classmethod
    def from_str(cls, value: str) -> "Engine":
        """Parse an engine name from a string (case-insensitive).

        Raises:
            ValueError: if the string doesn't match a known engine.
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unsupported LLM engine: {value!r}. "
            f"Supported: {[e.value for e in cls]}"
        )

    @property
    def is_local_server(self) -> bool:
        """True if this engine requires launching a local vLLM/Aphrodite server.

        Engines with is_local_server=True are launched via llm_server_context
        (which spawns a subprocess and waits for it to be healthy). Engines
        with is_local_server=False (e.g. OAI) use a pre-existing API endpoint.
        """
        return self in (Engine.VLLM, Engine.APHRODITE)

    @property
    def is_api_model(self) -> bool:
        """True if this engine is a hosted API model (not a local server).

        API models use different sampling-param semantics: they don't accept
        temperature/top_p/top_k in the same way, and disable_thinking is
        translated to reasoning_effort="none" instead of a chat_template_kwarg.
        """
        return self == Engine.OAI

    @property
    def uses_tokenizer(self) -> bool:
        """True if this engine requires a HuggingFace tokenizer for input length checking.

        Local server engines (vLLM, Aphrodite) tokenize inputs to check
        against max_input_tokens. API models use a simpler word-count heuristic
        against max_dataset_words.
        """
        return self.is_local_server

    @property
    def venv_env_var(self) -> str:
        """Environment variable name for the engine's virtualenv path.

        Local server engines are launched from their own venv (to avoid
        dependency conflicts with the main project venv).
        """
        if self == Engine.VLLM:
            return "VLLM_VENV_PATH"
        if self == Engine.APHRODITE:
            return "APHRODITE_VENV_PATH"
        raise ValueError(f"{self} does not use a virtualenv (only local server engines do).")

    @property
    def venv_default(self) -> str | None:
        """Default venv path if the env var is unset (None means no default)."""
        if self == Engine.APHRODITE:
            return ".aphrodite"
        return None

    @property
    def serve_subcommand(self) -> str:
        """The subcommand used to launch the engine (e.g. "serve" for vLLM, "run" for Aphrodite)."""
        if self == Engine.VLLM:
            return "serve"
        if self == Engine.APHRODITE:
            return "run"
        raise ValueError(f"{self} does not have a serve subcommand.")
