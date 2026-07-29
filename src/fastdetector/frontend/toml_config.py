from pydantic import BaseModel, model_validator
from typing import Any, Optional, List, Union

from fastdetector.frontend.engine_config import EngineConfig


class GlobalsConfig(BaseModel):
    """Global configuration settings for dataset processing, paths, and naming conventions."""

    # Dataset Overrides (if set, these bypass the prefix/suffix logic).
    # None means "not set"; use the prefix+suffix scheme instead.
    override_dataset_input: Optional[str] = None
    override_dataset_output: Optional[str] = None

    # Standard Naming Convention (Prefix + Suffix).
    dataset_prefix: str
    raw_suffix: str
    pre_filter_suffix: str
    post_filter_suffix: str
    gen_suffix: str
    stat_suffix: str
    eval_suffix: str

    # Engine Virtual Environment Paths
    vllm_venv_path: str = ".vllm"
    aphrodite_venv_path: str = ".aphrodite"

    def resolve_input_dataset(self, suffix: str) -> str:
        """Return the source dataset name for the given suffix, honouring override_dataset_input.

        Args:
            suffix: Dataset suffix string.

        Returns:
            Resolved input dataset name.
        """
        if self.override_dataset_input is not None:
            return self.override_dataset_input
        return f"{self.dataset_prefix}-{suffix}"

    def resolve_output_dataset(self, suffix: str) -> str:
        """Return the target dataset name for the given suffix, honouring override_dataset_output.

        Args:
            suffix: Dataset suffix string.

        Returns:
            Resolved output dataset name.
        """
        if self.override_dataset_output is not None:
            return self.override_dataset_output
        return f"{self.dataset_prefix}-{suffix}"


class ConditionConfig(BaseModel):
    column: str
    operator: str
    value: Any

    
class PipeConfig(BaseModel):
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

    # Server batching (local engines only). Defaults match the previously
    # hardcoded values.
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
    name: str
    suffix: str
    direction: str = "higher_is_ai"
    # Which of the global threshold settings apply to this classifier:
    # "score" -> threshold_type_score / manual_threshold_score
    # "bin"   -> threshold_type_bin / manual_threshold_bin
    threshold_kind: str = "score"

    @model_validator(mode="after")
    def _check_threshold_kind(self):
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
    # Each classifier selects its settings via threshold_kind ("score"/"bin").
    # Set the matching manual_threshold_* to a value to skip the validation
    # sweep for that classifier and use the fixed threshold instead.
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

    # Per-classifier bin breakdown: every row is placed in one of num_bins
    # equal-count (quantile) bins of bin_column, and each classifier is scored
    # per bin. This is what shows whether detection only works on the heavily
    # rewritten rows. bin_column defaults to the first distance metric.
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

    # Sequence-length caps for the Qwen3 embedding/reranker passes. None
    # inherits the checkpoint's own limit (40960 tokens), which lets a single
    # runaway generation set the memory cost of its whole batch, since
    # SentenceTransformer.encode sorts by length and pads to the longest
    # member. Note these change metric values for texts longer than the cap:
    # the tail beyond it is not seen by the model.
    embedding_max_seq_length: Optional[int] = None
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
    """Configuration for exact LLM-based metric extraction (llm_stats.py).

    Models are loaded in-process via transformers; metrics are computed from
    exact full-vocabulary distributions with fused reductions (no logprobs
    are stored). Batch-id sharding splits the dataset across machines (one
    shard per run, as elsewhere in the pipeline); within a run, the
    checkpoint(s) are replicated onto every configured GPU and all replicas
    work that single shard in parallel.
    """
    columns_to_score: List[str]

    # Maximum tokens per text; longer texts are truncated.
    max_model_len: int = 16000

    # Cap on padded tokens (batch_size * max_len) per forward pass. Texts are
    # length-sorted into buckets so padding waste stays low.
    max_batch_tokens: int = 16384

    # Positions per LM-head/log-softmax reduction chunk. Peak activation
    # memory scales with head_chunk_size * vocab_size (per co-resident model).
    head_chunk_size: int = 512

    # Model dtype: "bfloat16", "float16", or "float32". Reductions always run
    # in float32.
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
    binoculars_score: bool
    fastdetectgpt_score: bool

    topp_threshold: float = 0.95
    topk_threshold: int = 50

    # Model checkpoints and their aligned output-column suffixes. With
    # binoculars enabled, the first checkpoint is the observer and the second
    # the performer, and both must share a tokenizer/vocab.
    llm_checkpoints: List[str]
    col_suffixes: List[str]

    @model_validator(mode='after')
    def validate_llm_settings(self) -> 'LLMStatConfig':
        """Validate alignment of LLM checkpoints, column suffixes, and binoculars configuration.

        Returns:
            Self instance if validation succeeds.

        Raises:
            ValueError: If llm_checkpoints and col_suffixes lengths differ or binoculars requirements are unmet.
        """
        if len(self.llm_checkpoints) != len(self.col_suffixes):
            raise ValueError(f"Length mismatch: llm_checkpoints ({len(self.llm_checkpoints)}) must match col_suffixes ({len(self.col_suffixes)})")
        if len(set(self.col_suffixes)) != len(self.col_suffixes):
            raise ValueError(f"col_suffixes must be unique, got {self.col_suffixes}")
        if self.binoculars_score and len(self.llm_checkpoints) != 2:
            raise ValueError(f"Binoculars score requires exactly 2 llm_checkpoints, but {len(self.llm_checkpoints)} were provided.")
        return self


class EditLensStatConfig(BaseModel):
    """Configuration for EditLens-specific bucket and score inference (editlens_stats.py)."""
    columns_to_score: List[str]
    suffix: str
    
    base_model: str
    checkpoint: str
    max_length: int
    batch_size: int
