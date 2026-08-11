import os

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.providers.objects import (
    BatchProvider,
    BatchResult,
    BatchState,
)

__all__ = ["BatchProvider", "BatchResult", "BatchState", "make_provider"]


def make_provider(pipe_config) -> BatchProvider:
    """Construct the batch provider for a pipeline configuration.

    Args:
        pipe_config: A PipeConfig whose engine is a hosted API.

    Returns:
        A BatchProvider bound to the configured transport.

    Raises:
        ValueError: if the engine has no offline-batch transport.
    """
    engine = pipe_config.engine
    api_key = (
        os.environ.get(pipe_config.api_key_env, "")
        if pipe_config.api_key_env
        else ""
    )

    if engine == EngineConfig.OAI:
        from fastdetector.providers.openai_batch import OpenAIBatchProvider

        return OpenAIBatchProvider(api_key=api_key, base_url=pipe_config.api_url)

    if engine == EngineConfig.ANTHROPIC:
        from fastdetector.providers.anthropic_batch import AnthropicBatchProvider

        return AnthropicBatchProvider(api_key=api_key)

    if engine == EngineConfig.ANTHROPIC_AWS:
        from fastdetector.providers.anthropic_batch import AnthropicBatchProvider

        return AnthropicBatchProvider(
            use_aws=True,
            aws_region=pipe_config.aws_region or os.environ.get("AWS_REGION"),
        )

    raise ValueError(f"Engine {engine.value!r} has no offline-batch transport.")
