"""Configuration consumed by :func:`fastdetector.frontend.pipe.run_pipeline`.

A ``PipeConfig`` carries everything ``run_pipeline`` needs to drive a
generation pass: the engine/model/sampling parameters plus the dataset-shaping
knobs (``num_samples``, ``source_column``, ``prompt_file``).

Both ``gen.toml`` and ``filter.toml`` produce a ``PipeConfig``; ``filter.toml``
additionally supplies filter-specific extras handled by
:class:`fastdetector.frontend.config.FilterConfig`.
"""

from typing import Any, Optional

from pydantic import BaseModel, model_validator

from fastdetector.engine import Engine


class PipeConfig(BaseModel):
    """Configuration for the LLM generation pipeline.

    TOML layout (gen.toml, and the shared portion of filter.toml)::

        num_samples = 50000
        source_column = "..."
        prompt_file = "..."

        [pipeline]
        engine = "vllm"
        model_name = "..."
        temperature = 0.6
        ...

    The ``[pipeline]`` sub-section is flattened automatically by the
    ``_flatten_pipeline_section`` validator, so callers can pass the raw
    TOML dict directly: ``PipeConfig(**tomllib.load(f))``.
    """

    # Engine
    engine: Engine
    model_name: str

    # Sampling parameters.
    # top_k / disable_thinking are optional because filter.toml legitimately
    # omits them (filtering uses temperature=0 and ignores these knobs).
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    disable_thinking: Optional[bool] = None
    presence_penalty: float

    # Aphrodite-specific sampling parameters
    top_a: float = 0.0
    xtc_probability: float = 0.0
    nsigma: float = 0.0

    # Length limits
    max_model_len: Optional[int] = None
    max_input_tokens: Optional[int] = None
    max_dataset_words: Optional[int] = None

    # API settings
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None

    # Dataset shaping
    num_samples: int
    source_column: str
    prompt_file: str

    @model_validator(mode="before")
    @classmethod
    def _flatten_pipeline_section(cls, data: Any) -> Any:
        """Flatten the TOML ``[pipeline]`` sub-section into the top level.

        gen.toml and filter.toml both group engine/sampling fields under a
        ``[pipeline]`` table while keeping ``num_samples``/``source_column``/
        ``prompt_file`` at the top level. This validator flattens that layout
        so callers can pass the raw TOML dict directly:
        ``PipeConfig(**tomllib.load(f))``.

        Raises:
            ValueError: if the input is not a dict or has no ``[pipeline]``
                section. The flat-dict layout (engine/model_name at the top
                level) is no longer supported.
        """
        if not isinstance(data, dict):
            raise ValueError(
                "PipeConfig must be initialized from a dict with a [pipeline] section; "
                f"got {type(data).__name__}."
            )
        if "pipeline" not in data:
            raise ValueError(
                "PipeConfig requires a [pipeline] section in the TOML config "
                "(engine, model_name, temperature, etc. must live under [pipeline], "
                "not at the top level)."
            )
        pipeline = dict(data["pipeline"])
        flat = {k: v for k, v in data.items() if k != "pipeline"}
        flat.update(pipeline)
        return flat
