from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class GlobalsConfig(BaseModel):
    """Global configuration settings for dataset processing, paths, and naming conventions."""
    
    # Dataset Overrides (if set, these bypass the prefix/suffix logic)
    override_dataset_input: str
    override_dataset_output: str
    
    # Standard Naming Convention (Prefix + Suffix)
    dataset_prefix: str
    raw_suffix: str
    pre_filter_suffix: str
    post_filter_suffix: str
    gen_suffix: str
    stat_suffix: str
    eval_suffix: str
    
    # Execution & Storage Flags
    save_locally_instead: bool
    periodic_checkpoint: bool
    cache_dir: str

class PipelineConfig(BaseModel):
    """Configuration for the LLM generation pipeline and engine parameters."""
    
    # Core Engine Settings
    engine: str
    model_name: str
    
    # Sampling Parameters
    temperature: float
    top_p: float
    top_k: int
    disable_thinking: bool
    presence_penalty: float
    
    # Length Limits (Optional)
    max_model_len: Optional[int] = None
    max_input_tokens: Optional[int] = None
    
    # Aphrodite-specific Sampling Parameters
    top_a: float = 0.0
    xtc_probability: float = 0.0
    nsigma: float = 0.0
    
    # API & General Settings
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None
    max_dataset_words: Optional[int] = None

class GenConfig(BaseModel):
    """Configuration for the generation scripts (gen.py)."""
    num_samples: int
    source_column: str
    prompt_file: str
    pipeline: PipelineConfig

class FilterConfig(BaseModel):
    """Configuration for the filtering script (filter.py)."""
    num_samples: int
    source_column: str
    prompt_file: str
    output_shards: int
    conditions: str
    pipeline: PipelineConfig

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
    
    # Thresholds & Splits
    validation_size: float
    threshold_type_bin: str
    threshold_type_score: str
    manual_threshold_score: float
    manual_threshold_bin: float
    
    # Distance Metrics
    distance_metric_filter_type: str
    distance_metrics_lower_bounds: List[float]
    distance_metrics: List[str]

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
