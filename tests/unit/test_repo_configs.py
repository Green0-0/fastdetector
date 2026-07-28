"""The configs and prompt sets actually committed to this repository.

Nothing here is a hypothetical: every assertion runs against the real files a
pipeline run would load, so a typo'd operator or a renamed prompt file fails in
CI in a second instead of twenty minutes into a GPU job.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.toml_config import (
    AnalysisConfig,
    DistanceStatConfig,
    EditLensStatConfig,
    FilterConfig,
    GenConfig,
    GlobalsConfig,
    LLMStatConfig,
)
from fastdetector.frontend.toml_loader import load_toml
from fastdetector.prompting.prompts import load_prompts
from fastdetector.utils import apply_filter_conditions

SUPPORTED_OPERATORS = {"==", "!=", ">", "<", ">=", "<="}

STAGE_CONFIGS = {
    "config/filter.toml": FilterConfig,
    "config/analysis.toml": AnalysisConfig,
    "config/distance_stats.toml": DistanceStatConfig,
    "config/editlens_stats.toml": EditLensStatConfig,
    "config/llm_stats.toml": LLMStatConfig,
}


def gen_config_paths(repo_root):
    """Every generation config in ``config/gen``."""
    return sorted((repo_root / "config" / "gen").glob("*.toml"))


def prompt_paths(repo_root):
    """Every prompt set in ``prompts/``."""
    return sorted((repo_root / "prompts").glob("*.json"))


def pytest_generate_tests(metafunc):
    """Parameterise over the repo's config and prompt files."""
    if "gen_config_path" in metafunc.fixturenames:
        paths = gen_config_paths(REPO_ROOT)
        metafunc.parametrize("gen_config_path", paths, ids=[p.name for p in paths])
    if "prompt_path" in metafunc.fixturenames:
        paths = prompt_paths(REPO_ROOT)
        metafunc.parametrize("prompt_path", paths, ids=[p.name for p in paths])


# --------------------------------------------------------------------------
# TOML validity
# --------------------------------------------------------------------------


def test_every_committed_toml_parses(repo_root):
    paths = sorted((repo_root / "config").rglob("*.toml"))
    assert paths
    for path in paths:
        with path.open("rb") as handle:
            tomllib.load(handle)


def test_globals_config_validates(repo_root):
    config = GlobalsConfig(**load_toml(str(repo_root / "config" / "globals.toml")))
    assert config.dataset_prefix
    assert not config.dataset_prefix.endswith("-")


def test_globals_suffixes_are_distinct(repo_root):
    # Two stages sharing a suffix means one stage overwrites the other's
    # dataset on the Hub.
    config = GlobalsConfig(**load_toml(str(repo_root / "config" / "globals.toml")))
    suffixes = [
        config.raw_suffix,
        config.pre_filter_suffix,
        config.post_filter_suffix,
        config.gen_suffix,
        config.stat_suffix,
        config.eval_suffix,
    ]
    assert len(set(suffixes)) == len(suffixes)


@pytest.mark.parametrize(("relative_path", "model"), sorted(STAGE_CONFIGS.items()))
def test_stage_config_validates(repo_root, relative_path, model):
    config = model(**load_toml(str(repo_root / relative_path)))
    assert config is not None


def test_gen_config_validates(gen_config_path):
    config = GenConfig(**load_toml(str(gen_config_path)))
    assert config.source_column
    assert config.pipeline.model_name


# --------------------------------------------------------------------------
# Engine / sampling coherence
# --------------------------------------------------------------------------


def test_local_engine_configs_only_set_supported_sampling_params(
    repo_root, gen_config_path
):
    # Params an engine does not accept are silently dropped at run time; for
    # the local engines there is no reason to have any.
    config = GenConfig(**load_toml(str(gen_config_path)))
    engine = config.pipeline.engine
    if not engine.is_local_server:
        pytest.skip(f"{engine.value} intentionally ignores most sampling params")

    all_params = [
        "temperature",
        "top_p",
        "top_k",
        "presence_penalty",
        "disable_thinking",
        "top_a",
        "xtc_probability",
        "nsigma",
    ]
    unsupported = [
        name
        for name in all_params
        if getattr(config.pipeline, name, None) is not None
        and name not in engine.valid_sampling_params
    ]
    assert unsupported == []


def test_api_engine_configs_declare_an_endpoint(gen_config_path):
    config = GenConfig(**load_toml(str(gen_config_path)))
    if config.pipeline.engine.is_local_server:
        pytest.skip("local engines get their URL from the launched server")
    assert config.pipeline.api_url, "an API engine needs api_url"
    assert config.pipeline.api_key_env, "an API engine needs api_key_env"


def test_filter_config_engine_is_coherent(repo_root):
    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    if not config.pipeline.engine.is_local_server:
        assert config.pipeline.api_url


def test_input_length_limit_fits_the_context_window(repo_root, gen_config_path):
    config = GenConfig(**load_toml(str(gen_config_path)))
    pipeline = config.pipeline
    if pipeline.max_input_len is None or pipeline.max_model_len is None:
        pytest.skip("no explicit limits configured")
    # The prompt template and the answer both have to fit alongside the input.
    assert pipeline.max_input_len < pipeline.max_model_len


# --------------------------------------------------------------------------
# Filter conditions
# --------------------------------------------------------------------------


def test_filter_conditions_use_supported_operators(repo_root):
    # An unrecognised operator is not an error at run time: every row simply
    # evaluates to False and the filtered dataset comes out empty.
    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    for condition in config.conditions:
        assert condition.operator in SUPPORTED_OPERATORS


def test_analysis_filter_conditions_use_supported_operators(repo_root):
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    for condition in config.filter_conditions:
        assert condition.operator in SUPPORTED_OPERATORS


def test_filter_config_langdetect_threshold_is_a_probability(repo_root):
    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    if config.langdetect_threshold is not None:
        assert 0.0 <= config.langdetect_threshold <= 1.0


def test_committed_filter_conditions_run_against_a_matching_dataset(repo_root):
    """The real conditions must actually select rows from a plausible dataset."""
    from datasets import Dataset

    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    row = {}
    for condition in config.conditions:
        if isinstance(condition.value, bool):
            row[condition.column] = condition.value
        elif condition.operator in {"<", "<="}:
            row[condition.column] = float(condition.value) - 0.1
        else:
            row[condition.column] = float(condition.value) + 0.1

    dataset = Dataset.from_list([row])
    kept = apply_filter_conditions(dataset, config.conditions, config.filter_type)
    assert len(kept) == 1


# --------------------------------------------------------------------------
# Analysis config
# --------------------------------------------------------------------------


def test_analysis_classifier_suffixes_are_unique(repo_root):
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    suffixes = [clf.suffix for clf in config.classifiers]
    assert len(set(suffixes)) == len(suffixes)


def test_analysis_classifier_names_are_unique(repo_root):
    # The summary JSON is keyed by name, so duplicates would overwrite.
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    names = [clf.name for clf in config.classifiers]
    assert len(set(names)) == len(names)


def test_analysis_class_definition_is_complete(repo_root):
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    if config.fixed_classes is not None:
        assert len(config.fixed_classes) == len(config.base_columns)
        assert any(config.fixed_classes), "no AI class configured"
        assert not all(config.fixed_classes), "no human class configured"
    else:
        assert config.auto_class_column, "need fixed_classes or auto_class_column"
        assert config.ai_label, "auto_class_column needs an ai_label"


def test_analysis_threshold_types_are_known(repo_root):
    from fastdetector.visualization.metrics import FPR_TARGETS

    valid = {"accuracy", "f1", *FPR_TARGETS}
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    assert config.threshold_type_score in valid
    assert config.threshold_type_bin in valid


def test_analysis_validation_size_is_a_fraction(repo_root):
    config = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    assert 0.0 < config.validation_size < 1.0


# --------------------------------------------------------------------------
# Stat configs
# --------------------------------------------------------------------------


def test_llm_stats_scores_the_columns_the_pipeline_produces(repo_root):
    globals_config = GlobalsConfig(**load_toml(str(repo_root / "config" / "globals.toml")))
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    assert config.columns_to_score
    assert globals_config.stat_suffix  # the dataset llm_stats reads and writes


def test_llm_stats_dtype_is_supported(repo_root):
    from fastdetector.statistics.exact_scorer import _resolve_dtype

    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    _resolve_dtype(config.dtype)


def test_llm_stats_batch_budget_admits_a_full_length_text(repo_root):
    # max_batch_tokens below max_model_len means the longest texts each get a
    # batch of one, which is legal but usually a config mistake.
    config = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    assert config.max_batch_tokens >= config.max_model_len


def test_stat_configs_agree_on_the_text_columns(repo_root):
    distance = DistanceStatConfig(
        **load_toml(str(repo_root / "config" / "distance_stats.toml"))
    )
    llm = LLMStatConfig(**load_toml(str(repo_root / "config" / "llm_stats.toml")))
    editlens = EditLensStatConfig(
        **load_toml(str(repo_root / "config" / "editlens_stats.toml"))
    )
    pair = {distance.human_column, distance.ai_column}
    assert pair <= set(llm.columns_to_score)
    assert pair <= set(editlens.columns_to_score)


def test_analysis_base_columns_match_the_scored_columns(repo_root):
    # analysis.py reads "<base_column><classifier suffix>", so the base columns
    # have to be the ones the stats stage actually scored.
    analysis = AnalysisConfig(**load_toml(str(repo_root / "config" / "analysis.toml")))
    editlens = EditLensStatConfig(
        **load_toml(str(repo_root / "config" / "editlens_stats.toml"))
    )
    assert set(analysis.base_columns) <= set(editlens.columns_to_score)


# --------------------------------------------------------------------------
# Prompt sets
# --------------------------------------------------------------------------


def test_prompt_file_loads(prompt_path):
    prompts = load_prompts([str(prompt_path)])
    assert prompts


def test_prompt_turns_are_non_empty(prompt_path):
    for index, prompt in enumerate(load_prompts([str(prompt_path)])):
        assert prompt.chat_turns, f"prompt {index} has no turns"
        assert all(turn.strip() for turn in prompt.chat_turns)


def test_first_turn_references_the_document(prompt_path):
    # {{DOC}} is substituted by PromptSet.map; without it the sample text never
    # reaches the model.
    for index, prompt in enumerate(load_prompts([str(prompt_path)])):
        assert "{{DOC}}" in prompt.chat_turns[0], f"prompt {index} ignores the document"


def test_response_placeholders_reference_earlier_turns(prompt_path):
    # {{RESP_n}} is only substituted for n < the current turn index; a forward
    # reference would be sent to the model verbatim.
    pattern = re.compile(r"\{\{RESP_(\d+)\}\}")
    for index, prompt in enumerate(load_prompts([str(prompt_path)])):
        for turn_index, turn in enumerate(prompt.chat_turns):
            for match in pattern.finditer(turn):
                assert int(match.group(1)) < turn_index, (
                    f"prompt {index} turn {turn_index} references {match.group(0)}"
                )


def test_prompt_metadata_declares_a_type(prompt_path):
    # analysis.py groups its per-prompt breakdown by PROMPT_TYPE.
    for prompt in load_prompts([str(prompt_path)]):
        assert prompt.metadata.get("PROMPT_TYPE")


def test_prompt_examples_are_user_assistant_pairs(prompt_path):
    for prompt in load_prompts([str(prompt_path)]):
        for example in prompt.examples:
            assert len(example) == 2
            assert all(isinstance(part, str) for part in example)


def test_prompt_files_are_valid_json_lists(prompt_path):
    data = json.loads(prompt_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)


# --------------------------------------------------------------------------
# Cross-references between configs and files
# --------------------------------------------------------------------------


def test_gen_configs_reference_an_existing_prompt_file(repo_root, gen_config_path):
    config = GenConfig(**load_toml(str(gen_config_path)))
    assert (repo_root / config.prompt_file).is_file(), config.prompt_file


def test_filter_config_references_an_existing_prompt_file(repo_root):
    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    assert (repo_root / config.prompt_file).is_file(), config.prompt_file


def test_numbered_gen_configs_are_contiguous(repo_root):
    # Batch ids are handed out as 0..N-1; a gap means one shard never runs.
    indices = sorted(
        int(path.stem.split("_")[1])
        for path in gen_config_paths(repo_root)
        if path.stem.split("_")[1].isdigit()
    )
    assert indices == list(range(len(indices)))


def test_all_numbered_gen_configs_agree_on_the_source_column(repo_root):
    columns = {
        GenConfig(**load_toml(str(path))).source_column
        for path in gen_config_paths(repo_root)
    }
    assert len(columns) == 1, f"gen shards disagree on the source column: {columns}"
