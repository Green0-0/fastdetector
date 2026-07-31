import pytest
from pydantic import ValidationError

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.toml_config import (
    AnalysisConfig,
    ClassifierConfig,
    ConditionConfig,
    DistanceStatConfig,
    EditLensStatConfig,
    FilterConfig,
    GenConfig,
    GlobalsConfig,
    LLMStatConfig,
    PipeConfig,
)

GLOBALS_FIELDS = {
    "dataset_prefix": "user/base-",
    "raw_dataset": "raw",
    "pre_filter_dataset": "processed",
    "post_filter_dataset": "filtered",
    "gen_dataset": "rewritten",
    "stat_dataset": "stat",
    "eval_dataset": "eval",
}

PIPELINE_FIELDS = {"engine": "vllm", "model_name": "some/model"}


def make_globals(**overrides) -> GlobalsConfig:
    """Build a GlobalsConfig with all required fields filled in."""
    return GlobalsConfig(**{**GLOBALS_FIELDS, **overrides})


def make_llm_stat_config(**overrides) -> LLMStatConfig:
    """Build an LLMStatConfig with all required fields filled in."""
    base = {
        "columns_to_score": ["original"],
        "perplexity": True,
        "entropy": True,
        "topp_outlier": True,
        "topk_outlier": True,
        "binoculars_score": False,
        "fastdetectgpt_score": True,
        "llm_checkpoints": ["a/model"],
        "col_suffixes": ["_a"],
    }
    return LLMStatConfig(**{**base, **overrides})


# --------------------------------------------------------------------------
# GlobalsConfig
# --------------------------------------------------------------------------


def test_globals_requires_every_dataset_field():
    with pytest.raises(ValidationError):
        GlobalsConfig(dataset_prefix="user/base-", raw_dataset="raw")


def test_globals_venv_defaults():
    config = make_globals()
    assert config.vllm_venv_path == ".vllm"
    assert config.aphrodite_venv_path == ".aphrodite"


def test_resolve_dataset_with_prefix():
    config = make_globals()
    assert config.resolve_dataset(config.raw_dataset) == "user/base-raw"
    assert config.resolve_dataset(config.stat_dataset) == "user/base-stat"


def test_resolve_dataset_without_prefix():
    config = make_globals(dataset_prefix="")
    assert config.resolve_dataset("user/base-raw") == "user/base-raw"


# --------------------------------------------------------------------------
# PipeConfig
# --------------------------------------------------------------------------


def test_pipe_config_coerces_engine_string_to_enum():
    config = PipeConfig(**PIPELINE_FIELDS)
    assert config.engine is EngineConfig.VLLM


def test_pipe_config_sampling_params_default_to_none():
    config = PipeConfig(**PIPELINE_FIELDS)
    for field in (
        "temperature",
        "top_p",
        "top_k",
        "disable_thinking",
        "presence_penalty",
        "top_a",
        "xtc_probability",
        "nsigma",
        "max_model_len",
        "max_input_len",
        "api_url",
        "api_key_env",
    ):
        assert getattr(config, field) is None, field


def test_pipe_config_server_batching_defaults():
    config = PipeConfig(**PIPELINE_FIELDS)
    assert config.max_num_seqs == 256
    assert config.max_num_batched_tokens == 2048
    assert config.parallelization_type == "data"


def test_pipe_config_rejects_unknown_engine():
    with pytest.raises(ValidationError):
        PipeConfig(engine="not-an-engine", model_name="m")


# --------------------------------------------------------------------------
# Gen / Filter
# --------------------------------------------------------------------------


def test_gen_config_roundtrip():
    config = GenConfig(
        source_column="text",
        prompt_file="prompts/p.json",
        pipeline=PIPELINE_FIELDS,
    )
    assert config.pipeline.engine is EngineConfig.VLLM
    assert config.source_column == "text"


def test_filter_config_defaults():
    config = FilterConfig(
        source_column="text",
        prompt_file="prompts/p.json",
        pipeline=PIPELINE_FIELDS,
    )
    assert config.conditions == []
    assert config.filter_type == "AND"
    assert config.langdetect_threshold is None


def test_filter_config_parses_conditions():
    config = FilterConfig(
        source_column="text",
        prompt_file="p.json",
        pipeline=PIPELINE_FIELDS,
        conditions=[{"column": "is_loose_subset", "operator": "==", "value": True}],
    )
    assert isinstance(config.conditions[0], ConditionConfig)
    assert config.conditions[0].value is True


def test_condition_value_is_untyped():
    assert ConditionConfig(column="c", operator="<", value=0.9).value == 0.9
    assert ConditionConfig(column="c", operator="==", value="s").value == "s"


# --------------------------------------------------------------------------
# ClassifierConfig / AnalysisConfig
# --------------------------------------------------------------------------


def test_classifier_defaults():
    clf = ClassifierConfig(name="EditLens", suffix="_editlens")
    assert clf.direction == "higher_is_ai"
    assert clf.threshold_kind == "score"


@pytest.mark.parametrize("kind", ["score", "bin"])
def test_classifier_accepts_valid_threshold_kinds(kind):
    assert ClassifierConfig(name="n", suffix="_s", threshold_kind=kind).threshold_kind == kind


def test_classifier_rejects_unknown_threshold_kind():
    with pytest.raises(ValidationError, match="threshold_kind"):
        ClassifierConfig(name="n", suffix="_s", threshold_kind="binn")


def test_analysis_config_defaults_and_nesting():
    config = AnalysisConfig(
        base_columns=["original", "final_response"],
        prompt_metadata_column="prompt",
        model_metadata_column="generator_model",
        validation_size=0.1,
        threshold_type_bin="f1",
        threshold_type_score="accuracy",
        classifiers=[{"name": "c", "suffix": "_c", "threshold_kind": "bin"}],
    )
    assert config.filter_type == "OR"
    assert config.distance_metrics == []
    assert config.manual_threshold_score is None
    assert isinstance(config.classifiers[0], ClassifierConfig)


# --------------------------------------------------------------------------
# LLMStatConfig validator
# --------------------------------------------------------------------------


def test_llm_stat_defaults():
    config = make_llm_stat_config()
    assert config.scorer.max_model_len == 16000
    assert config.scorer.max_batch_tokens == 16384
    assert config.scorer.head_chunk_size == 512
    assert config.scorer.dtype == "bfloat16"
    assert config.scorer.attn_implementation is None
    assert config.scorer.devices == "auto"
    assert config.scorer.topp_threshold == 0.95
    assert config.scorer.topk_threshold == 50


def test_llm_stat_reads_a_scorer_table():
    config = make_llm_stat_config(scorer={"max_model_len": 512, "dtype": "float32"})
    assert config.scorer.max_model_len == 512
    assert config.scorer.dtype == "float32"
    # Unset keys still fall back to the scorer's own defaults.
    assert config.scorer.head_chunk_size == 512


def test_llm_stat_rejects_checkpoint_suffix_length_mismatch():
    with pytest.raises(ValidationError, match="Length mismatch"):
        make_llm_stat_config(llm_checkpoints=["a", "b"], col_suffixes=["_a"])


def test_llm_stat_rejects_duplicate_suffixes():
    # Duplicate suffixes would make the second model silently overwrite the
    # first model's output columns.
    with pytest.raises(ValidationError, match="unique"):
        make_llm_stat_config(llm_checkpoints=["a", "b"], col_suffixes=["_a", "_a"])


def test_binoculars_requires_exactly_two_checkpoints():
    with pytest.raises(ValidationError, match="Binoculars"):
        make_llm_stat_config(binoculars_score=True)
    with pytest.raises(ValidationError, match="Binoculars"):
        make_llm_stat_config(
            binoculars_score=True,
            llm_checkpoints=["a", "b", "c"],
            col_suffixes=["_a", "_b", "_c"],
        )


def test_binoculars_accepts_two_checkpoints():
    config = make_llm_stat_config(
        binoculars_score=True, llm_checkpoints=["a", "b"], col_suffixes=["_a", "_b"]
    )
    assert config.binoculars_score is True


def test_devices_accepts_a_string_or_a_list():
    assert make_llm_stat_config(scorer={"devices": "auto"}).scorer.devices == "auto"
    assert make_llm_stat_config(
        scorer={"devices": ["cuda:0", "cuda:1"]}
    ).scorer.devices == ["cuda:0", "cuda:1"]


def test_cross_entropy_follows_the_binoculars_flag():
    # The scorer's cross-model term exists only to serve binoculars, so it is
    # derived rather than configured.
    assert make_llm_stat_config().scorer.compute_cross_entropy is False
    paired = make_llm_stat_config(
        binoculars_score=True, llm_checkpoints=["a", "b"], col_suffixes=["_a", "_b"]
    )
    assert paired.scorer.compute_cross_entropy is True


# --------------------------------------------------------------------------
# Stat configs
# --------------------------------------------------------------------------


def test_distance_stat_config_batch_size_defaults():
    config = DistanceStatConfig(
        human_column="original",
        ai_column="final_response",
        jaccard_1=True,
        jaccard_2=False,
        jaccard_3=False,
        levenshtein=True,
        moverscore=False,
        bertscore=False,
        cosdist=True,
        softngram=False,
        reranker=False,
        softngram_model="m1",
        embedding_model="m2",
        token_embedding_model="m3",
        reranker_model="m4",
    )
    assert config.embedding_batch_size == 4
    assert config.softngram_phrase_batch_size == 2048
    assert config.token_embedding_chunk_size == 100


def test_distance_stat_config_requires_all_metric_flags():
    with pytest.raises(ValidationError):
        DistanceStatConfig(human_column="a", ai_column="b")


def test_editlens_stat_config_requires_all_fields():
    config = EditLensStatConfig(
        columns_to_score=["original"],
        suffix="_rl",
        base_model="base",
        checkpoint="ckpt",
        max_length=512,
        batch_size=8,
    )
    assert config.columns_to_score == ["original"]
    with pytest.raises(ValidationError):
        EditLensStatConfig(columns_to_score=["original"], suffix="_rl")
