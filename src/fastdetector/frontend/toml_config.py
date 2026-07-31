from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, List, Union

from fastdetector.frontend.engine_config import EngineConfig


class GlobalsConfig(BaseModel):
    """Global configuration settings for dataset processing and paths."""

    # Optional dataset prefix
    dataset_prefix: str = ""

    # Dataset Name Paths
    raw_dataset: str
    pre_filter_dataset: str
    post_filter_dataset: str
    gen_dataset: str
    stat_dataset: str
    eval_dataset: str

    # Engine Virtual Environment Paths
    vllm_venv_path: str = ".vllm"
    aphrodite_venv_path: str = ".aphrodite"

    def resolve_dataset(self, dataset_path: str) -> str:
        """Return the resolved dataset repo ID by prepending dataset_prefix."""
        return f"{self.dataset_prefix}{dataset_path}"


class ConditionConfig(BaseModel):
    """Configuration for dataset filtering conditions based on column comparisons."""

    column: str
    operator: str
    value: Any


class PipeConfig(BaseModel):
    """Configuration for inference pipelines, sampling parameters, and engine options."""

    # Engine
    engine: EngineConfig
    model_name: str
    parallelization_type: str = "data"

    # Sampling parameters.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    disable_thinking: Optional[bool] = None
    presence_penalty: Optional[float] = None

    # Aphrodite-specific sampling parameters
    top_a: Optional[float] = None
    xtc_probability: Optional[float] = None
    nsigma: Optional[float] = None

    # Length limits
    max_model_len: Optional[int] = None
    max_input_len: Optional[int] = None

    # Engine batching (local engines only)
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048

    # API settings
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None


class GenConfig(BaseModel):
    """Configuration for the generation script (gen.py).

    How many rows a run covers is a property of the shard it reads, decided
    when the source dataset is sharded (scripts/shard_dataset.py), so there is
    no sample count here.
    """

    source_column: str
    prompt_file: str
    pipeline: PipeConfig


class FilterConfig(BaseModel):
    """Configuration for the filtering script (filter.py)."""

    source_column: str
    prompt_file: str
    pipeline: PipeConfig

    conditions: List[ConditionConfig] = []
    filter_type: str = "AND"

    # Langdetect threshold (0 to 1). If set, filter out rows where English prob < threshold.
    langdetect_threshold: Optional[float] = None


class ClassifierConfig(BaseModel):
    """Configuration for individual score/bin classifier evaluation settings."""

    name: str
    suffix: str
    direction: str = "higher_is_ai"
    # Which of the global threshold settings apply to this classifier:
    # "score" -> threshold_type_score / manual_threshold_score
    # "bin"   -> threshold_type_bin / manual_threshold_bin
    threshold_kind: str = "score"

    @model_validator(mode="after")
    def _check_threshold_kind(self):
        """Validate threshold_kind attribute to ensure it is 'score' or 'bin'.

        Returns:
            Self instance if valid.

        Raises:
            ValueError: If threshold_kind is not 'score' or 'bin'.
        """
        if self.threshold_kind not in ("score", "bin"):
            raise ValueError(
                f'threshold_kind must be "score" or "bin", got {self.threshold_kind!r}.'
            )
        return self

class AnalysisConfig(BaseModel):
    """Configuration for universal evaluation and subset breakdowns (analysis.py)."""

    # Global column context
    base_columns: List[str]
    fixed_classes: Optional[List[bool]] = None
    auto_class_column: Optional[str] = None
    ai_label: Optional[str] = None

    # Column Definitions
    prompt_metadata_column: str
    model_metadata_column: str

    # Thresholds & Splits.
    validation_size: float
    threshold_type_bin: str
    threshold_type_score: str
    manual_threshold_score: Optional[float] = None
    manual_threshold_bin: Optional[float] = None

    # Dataset Filtering
    filter_type: str = "OR"
    filter_conditions: List[ConditionConfig] = []

    # Distance Metrics for Correlation/Plots
    distance_metrics: List[str] = []

    # Classifiers to Evaluate
    classifiers: List[ClassifierConfig] = []

    bin_column: Optional[str] = None
    num_bins: int = 4


class DistanceStatConfig(BaseModel):
    """Configuration for distance-based statistics (distance_stats.py)."""
    human_column: str
    ai_column: str
    embedding_batch_size: int = 4
    token_embedding_batch_size: int = 4
    reranker_batch_size: int = 4
    softngram_phrase_batch_size: int = 2048
    token_embedding_chunk_size: int = 100

    # Sequence-length caps.
    embedding_max_seq_length: Optional[int] = None
    token_embedding_max_length: Optional[int] = None
    reranker_max_length: Optional[int] = None

    # Basic Similarity Metrics
    jaccard_1: bool
    jaccard_2: bool
    jaccard_3: bool
    levenshtein: bool

    # Deep Embedding Metrics
    moverscore: bool
    bertscore: bool
    cosdist: bool
    softngram: bool
    reranker: bool

    # Specific Models
    softngram_model: str
    embedding_model: str
    token_embedding_model: str
    reranker_model: str


class LLMStatConfig(BaseModel):
    """Configuration for exact LLM-based metric extraction (llm_stats.py)."""
    columns_to_score: List[str]

    # Nucleus probability mass threshold for top-p outlier detection.
    topp_threshold: float = Field(default=0.95, gt=0.0, le=1.0)

    # Rank threshold for top-k outlier detection.
    topk_threshold: int = Field(default=50, ge=1)

    # Maximum tokens per text; longer texts are truncated.
    max_model_len: int = Field(default=16000, ge=1)

    # Cap on padded tokens (batch_size * max_len) per forward pass.
    max_batch_tokens: int = Field(default=16384, ge=1)

    # Positions per LM-head/log-softmax reduction chunk. Peak activation memory
    # scales with head_chunk_size * vocab_size (per co-resident model).
    head_chunk_size: int = Field(default=512, ge=1)

    # Model dtype: any floating-point torch dtype name, e.g. "bfloat16".
    dtype: str = "bfloat16"

    # Attention backend. None tries flash_attention_2, then falls back to sdpa.
    attn_implementation: Optional[str] = None

    # Devices to score on. "auto" replicates the checkpoint(s) onto every
    # visible CUDA device (falling back to CPU); an explicit list such as
    # ["cuda:0", "cuda:1"] selects specific GPUs. With binoculars enabled,
    # each device holds both checkpoints.
    devices: Union[str, List[str]] = "auto"

    # LLM & Generation Metrics
    perplexity: bool
    entropy: bool
    topp_outlier: bool
    topk_outlier: bool
    binoculars: bool
    fastdetectgpt: bool

    # Model checkpoints and their aligned output-column suffixes
    llm_checkpoints: List[str]
    col_suffixes: List[str]

    @model_validator(mode="after")
    def validate_llm_settings(self) -> "LLMStatConfig":
        """Validate device, checkpoint, and suffix settings.

        Returns:
            Self instance if validation succeeds.

        Raises:
            ValueError: If devices is empty, llm_checkpoints and col_suffixes lengths differ, or binoculars requirements are unmet.
        """
        if isinstance(self.devices, str):
            if self.devices != "auto":
                self.devices = [self.devices]
        elif not self.devices:
            raise ValueError('devices must be "auto" or a non-empty list of device strings')
        if len(self.llm_checkpoints) != len(self.col_suffixes):
            raise ValueError(
                f"Length mismatch: llm_checkpoints ({len(self.llm_checkpoints)}) "
                f"must match col_suffixes ({len(self.col_suffixes)})"
            )
        if len(set(self.col_suffixes)) != len(self.col_suffixes):
            raise ValueError(f"col_suffixes must be unique, got {self.col_suffixes}")
        if self.binoculars and len(self.llm_checkpoints) != 2:
            raise ValueError(
                f"Binoculars score requires exactly 2 llm_checkpoints, "
                f"but {len(self.llm_checkpoints)} were provided."
            )
        return self


class EditLensStatConfig(BaseModel):
    """Configuration for EditLens-specific bucket and score inference (editlens_stats.py)."""

    columns_to_score: List[str]
    suffix: str

    base_model: str
    checkpoint: str
    max_length: int
    batch_size: int

