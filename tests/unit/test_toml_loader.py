"""TOML loading and the globals+stage config pairing used by every entry point."""

import tomllib

import pytest
from pydantic import ValidationError

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.toml_config import GenConfig, GlobalsConfig, LLMStatConfig
from fastdetector.frontend.toml_loader import load_config_pair, load_toml


def test_load_toml_returns_plain_dict(data_dir):
    parsed = load_toml(str(data_dir / "globals.toml"))
    assert isinstance(parsed, dict)
    assert parsed["dataset_prefix"] == "testuser/testds"


def test_load_toml_missing_file():
    with pytest.raises(FileNotFoundError):
        load_toml("does/not/exist.toml")


def test_load_toml_rejects_malformed_toml(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        load_toml(str(bad))


def test_load_config_pair_builds_both_models(data_dir):
    globals_config, gen_config = load_config_pair(
        str(data_dir / "globals.toml"), str(data_dir / "gen.toml"), GenConfig
    )
    assert isinstance(globals_config, GlobalsConfig)
    assert isinstance(gen_config, GenConfig)
    assert gen_config.pipeline.engine is EngineConfig.VLLM
    assert gen_config.source_column == "collected_subset"
    assert globals_config.resolve_output_dataset(globals_config.gen_suffix) == (
        "testuser/testds-rewritten"
    )


def test_load_config_pair_reads_overrides_and_venv_paths(data_dir):
    globals_config, _ = load_config_pair(
        str(data_dir / "globals_override.toml"), str(data_dir / "gen.toml"), GenConfig
    )
    assert globals_config.resolve_input_dataset("raw") == "someone/input-only"
    assert globals_config.vllm_venv_path == ".vllm-custom"
    assert globals_config.aphrodite_venv_path == ".aphrodite-custom"


def test_load_config_pair_surfaces_globals_validation_errors(data_dir):
    with pytest.raises(ValidationError):
        load_config_pair(
            str(data_dir / "globals_missing_field.toml"),
            str(data_dir / "gen.toml"),
            GenConfig,
        )


def test_load_config_pair_surfaces_stage_validation_errors(data_dir, tmp_path):
    stage = tmp_path / "llm_stats.toml"
    stage.write_text(
        "\n".join(
            [
                'columns_to_score = ["original"]',
                "perplexity = true",
                "entropy = true",
                "topp_outlier = true",
                "topk_outlier = true",
                "binoculars_score = true",
                "fastdetectgpt_score = true",
                'llm_checkpoints = ["only/one"]',
                'col_suffixes = ["_one"]',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Binoculars"):
        load_config_pair(str(data_dir / "globals.toml"), str(stage), LLMStatConfig)


def test_load_config_pair_is_generic_over_the_stage_class(data_dir, tmp_path):
    stage = tmp_path / "llm_stats.toml"
    stage.write_text(
        "\n".join(
            [
                'columns_to_score = ["original", "final_response"]',
                "perplexity = true",
                "entropy = false",
                "topp_outlier = false",
                "topk_outlier = false",
                "binoculars_score = false",
                "fastdetectgpt_score = false",
                'llm_checkpoints = ["a/b"]',
                'col_suffixes = ["_ab"]',
                'devices = ["cuda:0"]',
            ]
        ),
        encoding="utf-8",
    )
    _, config = load_config_pair(str(data_dir / "globals.toml"), str(stage), LLMStatConfig)
    assert isinstance(config, LLMStatConfig)
    assert config.devices == ["cuda:0"]
