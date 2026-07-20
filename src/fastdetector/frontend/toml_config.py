from pydantic import BaseModel
from typing import Any, Optional, List

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

    # Execution & Storage Flags
    save_locally_instead: bool
    cache_dir: str

    # Engine Virtual Environment Paths
    vllm_venv_path: str = ".vllm"
    aphrodite_venv_path: str = ".aphrodite"

    def resolve_input_dataset(self, suffix: str) -> str:
        """Return the source dataset name for the given suffix, honouring override_dataset_input."""
        if self.override_dataset_input is not None:
            return self.override_dataset_input
        return f"{self.dataset_prefix}-{suffix}"

    def resolve_output_dataset(self, suffix: str) -> str:
        """Return the target dataset name for the given suffix, honouring override_dataset_output."""
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

    # API settings
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None


class GenConfig(BaseModel):
    """Configuration for the generation script (gen.py)."""
    num_samples: int
    source_column: str
    prompt_file: str
    pipeline: PipeConfig


class FilterConfig(BaseModel):
    """Configuration for the filtering script (filter.py)."""
    num_samples: int
    source_column: str
    prompt_file: str
    pipeline: PipeConfig

    # Number of shards to split the filtered dataset into.
    # None means don't shard (upload as a single 'default' config).
    output_shards: Optional[int] = None
    conditions: List[ConditionConfig] = []
    filter_type: str = "AND"


class EvalConfig(BaseModel):
    """Configuration for downstream evaluation and classification thresholds."""

    # Column Definitions
    prompt_metadata_column: str
    model_metadata_column: str

    # Classifier Model Settings
    base_model: str
    checkpoint: str
    max_length: int
    batch_size: int

    # Thresholds & Splits.
    # Set manual_threshold_* to None to auto-derive from validation sweep.
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


class StatConfig(BaseModel):
    """Configuration for the comprehensive dataset statistics pipeline (stat.py)."""

    # Column Mapping
    human_column: str
    ai_column: str
    parallelization_type: str = "data"

    # Basic Similarity Metrics
    jaccards_1: bool
    jaccards_2: bool
    jaccards_3: bool
    levenshteins: bool

    # Deep Embedding Metrics
    moverscore: bool
    bertscore: bool
    pairwise_cosim: bool
    pairwise_softngram: bool

    # LLM & Generation Metrics
    perplexity: bool
    entropy: bool
    topp_outlier: bool
    topk_outlier: bool
    binoculars_score: bool
    fastdetectgpt_score: bool
    reranker_score: bool

    # Runtime & Optimization Flags
    threshold_type: str
    remove_columns_afterwards: bool
    batch_size: int

    # Specific Models
    softngram_model: str
    embedding_model: str
    token_embedding_model: str
    reranker_model: str

    # LLM Endpoints
    llm_checkpoints: List[str]
    col_suffixes: List[str]
    top_logprobs_k: int
