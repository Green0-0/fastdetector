import json
import types

import pytest

from fastdetector.frontend import pipe as pipe_module
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.frontend.toml_config import GlobalsConfig, PipeConfig

GLOBALS_FIELDS = {
    "dataset_prefix": "user/base-",
    "raw_dataset": "raw",
    "post_filter_dataset": "filtered",
    "gen_dataset": "rewritten",
    "stat_dataset": "stat",
    "eval_dataset": "eval",
}


class StubTokenizer:
    """Tokenizer stand-in whose token count is the whitespace word count."""

    @classmethod
    def from_pretrained(cls, model_name, *args, **kwargs):
        """Record nothing; just hand back an instance."""
        return cls()

    def encode(self, text: str) -> list[int]:
        """Return one id per whitespace-separated word."""
        return [0] * len(text.split())


@pytest.fixture
def pipeline_env(monkeypatch, data_dir):
    """Stub out the dataset, tokenizer, engine server, and generation call.

    Returns:
        A namespace whose ``calls`` dict records what the pipeline handed to
        each collaborator, plus knobs (``rows``, ``result``) for the stubs.
    """
    env = types.SimpleNamespace(
        calls={},
        rows=[{"text": "one two three"}],
        result=({"original": ["x"], "final_response": ["y"]}, 11, 22, 0),
        prompt_file=str(data_dir / "prompts_valid.json"),
    )

    def fake_load_dataset(dataset_name, split="train", subset_index=0):
        env.calls["load_dataset"] = {
            "dataset_name": dataset_name,
            "split": split,
            "subset_index": subset_index,
        }
        return env.rows

    def fake_build_dataset(samples, api_url, prompts, generation_params, **kwargs):
        env.calls["build_dataset"] = {
            "samples": list(samples),
            "api_url": api_url,
            "generation_params": generation_params,
            **kwargs,
        }
        return env.result

    class FakeServerContext:
        """Context manager stub simulating llm_server_context."""

        def __init__(self, **kwargs):
            env.calls["llm_server_context"] = kwargs

        def __enter__(self):
            return "http://localhost:9999/v1"

        def __exit__(self, *exc):
            env.calls["server_closed"] = True
            return False

    monkeypatch.setattr(pipe_module, "load_dataset_auto_shard", fake_load_dataset)
    monkeypatch.setattr(pipe_module, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(pipe_module, "llm_server_context", FakeServerContext)
    monkeypatch.setattr(pipe_module, "AutoTokenizer", StubTokenizer)
    return env


def run(env, *, engine="vllm", num_samples=10, source_column="text", **pipe_fields):
    """Invoke run_pipeline with a config built from the given overrides."""
    pipe_config = PipeConfig(engine=engine, model_name="some/model", **pipe_fields)
    return run_pipeline(
        globals_config=GlobalsConfig(**GLOBALS_FIELDS),
        pipe_config=pipe_config,
        prompt_file=env.prompt_file,
        num_samples=num_samples,
        source_column=source_column,
        source_dataset_name="user/base-filtered",
        batch_id=3,
    )


# --------------------------------------------------------------------------
# Sampling parameter translation
# --------------------------------------------------------------------------


def test_vllm_splits_params_between_the_body_and_extra_body(pipeline_env):
    run(
        pipeline_env,
        engine="vllm",
        temperature=0.6,
        top_p=0.9,
        presence_penalty=1.5,
        top_k=40,
    )
    params = pipeline_env.calls["build_dataset"]["generation_params"]
    assert params["temperature"] == 0.6
    assert params["top_p"] == 0.9
    assert params["presence_penalty"] == 1.5
    # top_k is not an OpenAI-API field, so it has to travel in extra_body.
    assert params["extra_body"] == {"top_k": 40}


def test_unset_params_are_not_sent_at_all(pipeline_env):
    run(pipeline_env, engine="vllm", temperature=0.6)
    params = pipeline_env.calls["build_dataset"]["generation_params"]
    assert params == {"temperature": 0.6}


def test_temperature_of_zero_is_still_sent(pipeline_env):
    # 0.0 is falsy but meaningful (greedy decoding); it must not be dropped.
    run(pipeline_env, engine="vllm", temperature=0.0)
    assert pipeline_env.calls["build_dataset"]["generation_params"]["temperature"] == 0.0


def test_disable_thinking_becomes_a_chat_template_kwarg_for_local_engines(pipeline_env):
    run(pipeline_env, engine="vllm", disable_thinking=True)
    params = pipeline_env.calls["build_dataset"]["generation_params"]
    assert params["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_disable_thinking_false_sends_nothing(pipeline_env):
    run(pipeline_env, engine="vllm", disable_thinking=False)
    assert pipeline_env.calls["build_dataset"]["generation_params"] == {}


def test_disable_thinking_becomes_reasoning_effort_for_proprietary_engines(
    pipeline_env,
):
    run(pipeline_env, engine="oai", disable_thinking=True, api_url="https://api/v1")
    params = pipeline_env.calls["build_dataset"]["generation_params"]
    assert params["reasoning_effort"] == "none"
    assert "extra_body" not in params


def test_aphrodite_only_params_reach_extra_body(pipeline_env):
    run(
        pipeline_env,
        engine="aphrodite",
        top_a=0.1,
        xtc_probability=0.5,
        nsigma=1.5,
    )
    extra = pipeline_env.calls["build_dataset"]["generation_params"]["extra_body"]
    assert extra == {"top_a": 0.1, "xtc_probability": 0.5, "nsigma": 1.5}


def test_params_unsupported_by_the_engine_are_dropped(pipeline_env):
    run(
        pipeline_env,
        engine="oai",
        api_url="https://api/v1",
        temperature=0.7,
        top_k=40,
        nsigma=1.0,
    )
    params = pipeline_env.calls["build_dataset"]["generation_params"]
    assert "temperature" not in params
    assert "extra_body" not in params


def test_dropped_params_are_reported_loudly(pipeline_env, capsys):
    run(pipeline_env, engine="vllm", temperature=0.7, top_a=0.2, nsigma=1.0)
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "top_a" in printed
    assert "nsigma" in printed
    assert "temperature" not in printed.split("WARNING", 1)[1]


def test_dropped_params_are_recorded_in_the_readme(pipeline_env):
    _, readme = run(pipeline_env, engine="vllm", top_a=0.2)
    assert "Ignored Params" in readme
    assert "top_a" in readme


def test_readme_says_none_when_nothing_was_dropped(pipeline_env):
    _, readme = run(pipeline_env, engine="vllm", temperature=0.5)
    assert "Ignored Params (unsupported by this engine): None" in readme


# --------------------------------------------------------------------------
# Sample selection
# --------------------------------------------------------------------------


def test_samples_are_read_from_the_configured_column_and_shard(pipeline_env):
    pipeline_env.rows = [{"text": "a b", "other": "ignored"}]
    run(pipeline_env, source_column="text")
    assert pipeline_env.calls["load_dataset"]["dataset_name"] == "user/base-filtered"
    assert pipeline_env.calls["load_dataset"]["subset_index"] == 3
    assert pipeline_env.calls["build_dataset"]["samples"] == ["a b"]


def test_sampling_stops_at_num_samples(pipeline_env):
    pipeline_env.rows = [{"text": f"row {i}"} for i in range(100)]
    run(pipeline_env, num_samples=5)
    assert len(pipeline_env.calls["build_dataset"]["samples"]) == 5


def test_no_num_samples_consumes_the_whole_shard(pipeline_env):
    # What the entrypoints do: the shard already holds exactly the rows this
    # run is meant to cover, so there is nothing left to cap.
    pipeline_env.rows = [{"text": f"row {i}"} for i in range(100)]
    run(pipeline_env, num_samples=None)
    assert len(pipeline_env.calls["build_dataset"]["samples"]) == 100


def test_rows_over_the_token_limit_are_dropped(pipeline_env):
    pipeline_env.rows = [
        {"text": "one two three four five"},
        {"text": "short"},
    ]
    run(pipeline_env, max_input_len=3)
    assert pipeline_env.calls["build_dataset"]["samples"] == ["short"]


def test_dropped_rows_do_not_count_towards_num_samples(pipeline_env):
    pipeline_env.rows = [{"text": "way too many words here"}] + [
        {"text": "ok"} for _ in range(4)
    ]
    run(pipeline_env, num_samples=4, max_input_len=2)
    assert len(pipeline_env.calls["build_dataset"]["samples"]) == 4


def test_the_drop_count_is_reported_in_the_readme(pipeline_env):
    pipeline_env.rows = [{"text": "a b c d"}, {"text": "a"}]
    _, readme = run(pipeline_env, max_input_len=2)
    assert "Dropped Samples (over length limit 2 tokens): 1" in readme


def test_without_a_length_limit_nothing_is_dropped(pipeline_env):
    pipeline_env.rows = [{"text": "a " * 500}]
    run(pipeline_env)
    assert len(pipeline_env.calls["build_dataset"]["samples"]) == 1


def test_null_cells_become_empty_strings(pipeline_env):
    pipeline_env.rows = [{"text": None}, {"text": 42}]
    run(pipeline_env)
    assert pipeline_env.calls["build_dataset"]["samples"] == ["", "42"]


def test_proprietary_engines_measure_length_in_words_not_tokens(pipeline_env):
    # There is no local tokenizer for an API model, so the limit is applied to
    # the word count and the readme has to say so.
    pipeline_env.rows = [{"text": "one two three four"}, {"text": "one two"}]
    _, readme = run(
        pipeline_env, engine="oai", api_url="https://api/v1", max_input_len=3
    )
    assert pipeline_env.calls["build_dataset"]["samples"] == ["one two"]
    assert "words" in readme


# --------------------------------------------------------------------------
# Engine wiring
# --------------------------------------------------------------------------


def test_local_engines_launch_a_server_and_use_its_url(pipeline_env):
    run(pipeline_env, engine="vllm", max_model_len=4096, max_num_seqs=8)
    launch = pipeline_env.calls["llm_server_context"]
    assert launch["model_name"] == "some/model"
    assert launch["venv_path"] == ".vllm"
    assert launch["max_model_len"] == 4096
    assert launch["max_num_seqs"] == 8
    assert launch["max_num_batched_tokens"] == 2048
    assert pipeline_env.calls["build_dataset"]["api_url"] == "http://localhost:9999/v1"
    assert pipeline_env.calls["server_closed"] is True


def test_aphrodite_uses_its_own_venv_path(pipeline_env):
    run(pipeline_env, engine="aphrodite")
    assert pipeline_env.calls["llm_server_context"]["venv_path"] == ".aphrodite"


def test_max_model_len_is_only_forwarded_when_configured(pipeline_env):
    run(pipeline_env, engine="vllm")
    assert "max_model_len" not in pipeline_env.calls["llm_server_context"]


def test_api_engines_use_the_configured_url_without_launching_anything(pipeline_env):
    run(pipeline_env, engine="oai", api_url="https://api.example/v1")
    assert "llm_server_context" not in pipeline_env.calls
    call = pipeline_env.calls["build_dataset"]
    assert call["api_url"] == "https://api.example/v1"
    assert call["model_name"] == "some/model"


def test_api_key_is_read_from_the_configured_environment_variable(
    pipeline_env, monkeypatch
):
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-secret")
    run(
        pipeline_env,
        engine="oai",
        api_url="https://api/v1",
        api_key_env="MY_PROVIDER_KEY",
    )
    assert pipeline_env.calls["build_dataset"]["api_key"] == "sk-secret"


def test_a_missing_api_key_variable_falls_back_to_empty(pipeline_env, monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    run(pipeline_env, engine="oai", api_url="https://api/v1", api_key_env="MISSING_KEY")
    assert pipeline_env.calls["build_dataset"]["api_key"] == "EMPTY"


def test_no_api_key_env_configured_falls_back_to_empty(pipeline_env):
    run(pipeline_env, engine="oai", api_url="https://api/v1")
    assert pipeline_env.calls["build_dataset"]["api_key"] == "EMPTY"


# --------------------------------------------------------------------------
# Result assembly
# --------------------------------------------------------------------------


def test_provenance_columns_are_added_for_every_row(pipeline_env):
    pipeline_env.result = (
        {"original": ["a", "b"], "final_response": ["c", "d"]},
        1,
        2,
        0,
    )
    dataset, _ = run(pipeline_env, engine="vllm", temperature=0.6)
    assert dataset.column_names == [
        "original",
        "final_response",
        "generator_model",
        "generation_params",
    ]
    assert dataset["generator_model"] == ["some/model", "some/model"]
    assert json.loads(dataset["generation_params"][0]) == {"temperature": 0.6}


def test_mismatched_column_lengths_are_rejected(pipeline_env):
    # Silently truncating here would misalign every row of the dataset.
    pipeline_env.result = ({"original": ["a", "b"], "final_response": ["c"]}, 0, 0, 0)
    with pytest.raises(RuntimeError, match="mismatched lengths"):
        run(pipeline_env)


def test_an_empty_result_produces_an_empty_dataset(pipeline_env):
    pipeline_env.result = ({}, 0, 0, 0)
    dataset, _ = run(pipeline_env)
    assert len(dataset) == 0


def test_readme_records_the_run(pipeline_env):
    pipeline_env.result = ({"original": ["a"], "final_response": ["b"]}, 111, 222, 4)
    _, readme = run(pipeline_env, engine="vllm", temperature=0.6)
    assert "Model Name: some/model" in readme
    assert "Source Dataset: user/base-filtered" in readme
    assert "Source Column: text" in readme
    assert "Failed API Requests: 4" in readme
    assert "Total Input Tokens Processed: 111" in readme
    assert "Total Output Tokens Processed: 222" in readme
    assert "Engine: vllm" in readme
    assert "Total Train Prompts: 2" in readme
