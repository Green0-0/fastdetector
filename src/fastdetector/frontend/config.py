from pydantic import BaseModel
from typing import Optional, List, Any

from fastdetector.frontend.pipeconfig import PipeConfig

class GlobalsConfig(BaseModel):
    """Global configuration settings for dataset processing, paths, and naming conventions."""

    # Dataset Overrides (if set, these bypass the prefix/suffix logic).
    # Empty string means "not set"; use the prefix+suffix scheme instead.
    override_dataset_input: str = ""
    override_dataset_output: str = ""

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

    def resolve_input_dataset(self, suffix: str) -> str:
        """Return the source dataset name for *suffix*, honouring override_dataset_input."""
        if self.override_dataset_input:
            return self.override_dataset_input
        return f"{self.dataset_prefix}-{suffix}"

    def resolve_output_dataset(self, suffix: str) -> str:
        """Return the target dataset name for *suffix*, honouring override_dataset_output."""
        if self.override_dataset_output:
            return self.override_dataset_output
        return f"{self.dataset_prefix}-{suffix}"


class ConditionConfig(BaseModel):
    column: str
    operator: str
    value: Any


class FilterConfig(PipeConfig):
    """Configuration for the filtering script (filter.py).

    Extends :class:`PipeConfig` with filter-specific fields. The ``PipeConfig``
    portion drives ``run_pipeline``; the extras below drive the post-pipeline
    filtering step.
    """
    output_shards: int
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
    # manual_threshold_* use None (not -1.0 sentinel) to mean "auto-derive
    # from validation sweep".
    validation_size: float
    threshold_type_bin: str
    threshold_type_score: str
    manual_threshold_score: Optional[float] = None
    manual_threshold_bin: Optional[float] = None

    # Dataset Filtering
    filter_type: str = "AND"
    filter_conditions: List[ConditionConfig] = []

    # Distance Metrics for Correlation/Plots
    distance_metrics: List[str] = []


class StatConfig(BaseModel):
    """Configuration for the comprehensive dataset statistics pipeline (stat.py)."""

    # Column Mapping
    human_column: str
    ai_column: str

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
