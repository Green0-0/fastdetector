import json
from types import SimpleNamespace

import pytest

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.toml_config import PipeConfig
from fastdetector.providers import BatchState
from fastdetector.providers.anthropic_batch import AnthropicBatchProvider
from fastdetector.providers.openai_batch import OpenAIBatchProvider


def msgs(*contents) -> list[dict[str, str]]:
    """Build an alternating user/assistant message list from raw contents."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(contents)
    ]


# --------------------------------------------------------------------------
# request bodies, captured from submit()
# --------------------------------------------------------------------------


def openai_bodies(inputs, generation_params, model_name="gpt-5-6", max_output_tokens=4096):
    """Capture the request bodies OpenAIBatchProvider.submit uploads."""
    uploaded = {}
    provider = object.__new__(OpenAIBatchProvider)
    provider.name = "oai"

    def create(file, purpose):
        uploaded["lines"] = file.read().decode("utf-8").strip().splitlines()
        return SimpleNamespace(id="f1")

    provider.client = SimpleNamespace(
        files=SimpleNamespace(create=create),
        batches=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="b1")),
    )
    provider.submit(inputs, generation_params, model_name, max_output_tokens)
    return [json.loads(line) for line in uploaded["lines"]]


def anthropic_requests(inputs, generation_params, model_name="claude-opus-5",
                       max_output_tokens=4096):
    """Capture the requests AnthropicBatchProvider.submit creates."""
    captured = {}
    provider = object.__new__(AnthropicBatchProvider)
    provider.name = "anthropic"

    def create(requests):
        captured["requests"] = requests
        return SimpleNamespace(id="b1")

    provider.client = SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(create=create))
    )
    provider.submit(inputs, generation_params, model_name, max_output_tokens)
    return captured["requests"]


def test_openai_body_carries_messages_and_params():
    body = openai_bodies([msgs("hi")], {"temperature": 0.7})[0]["body"]
    assert body["model"] == "gpt-5-6"
    assert body["messages"] == msgs("hi")
    assert body["temperature"] == 0.7
    assert body["max_completion_tokens"] == 4096


def test_openai_custom_ids_track_caller_order():
    records = openai_bodies([msgs("a"), msgs("b")], {})
    assert [r["custom_id"] for r in records] == ["req-0", "req-1"]
    assert records[0]["url"] == "/v1/chat/completions"


def test_anthropic_body_requires_max_tokens():
    with pytest.raises(ValueError, match="max_output_tokens is required"):
        anthropic_requests([msgs("hi")], {}, max_output_tokens=None)


def test_anthropic_body_rejects_sampling_params():
    with pytest.raises(ValueError, match="removed in the Claude 5 family"):
        anthropic_requests([msgs("hi")], {"temperature": 0.7})


def test_anthropic_body_keeps_thinking_config():
    params = anthropic_requests([msgs("hi")], {"thinking": {"type": "disabled"}})[0]["params"]
    assert params == {
        "model": "claude-opus-5",
        "max_tokens": 4096,
        "messages": msgs("hi"),
        "thinking": {"type": "disabled"},
    }


def test_anthropic_model_id_is_not_bedrock_prefixed():
    params = anthropic_requests([msgs("hi")], {})[0]["params"]
    assert not params["model"].startswith("anthropic.")


# --------------------------------------------------------------------------
# batch state
# --------------------------------------------------------------------------


def test_state_roundtrips_a_submitted_job(tmp_path):
    BatchState(str(tmp_path), "shard0").record(0, "job-abc", "anthropic_aws", 500)
    record = BatchState(str(tmp_path), "shard0").get(0)
    assert record["job_id"] == "job-abc"
    assert record["n_requests"] == 500


def test_state_keys_turns_separately(tmp_path):
    state = BatchState(str(tmp_path), "shard0")
    state.record(0, "job-0", "oai", 10)
    state.record(1, "job-1", "oai", 8)
    reloaded = BatchState(str(tmp_path), "shard0")
    assert reloaded.get(0)["job_id"] == "job-0"
    assert reloaded.get(1)["job_id"] == "job-1"


def test_state_is_absent_before_submission(tmp_path):
    assert BatchState(str(tmp_path), "shard0").get(0) is None


def test_corrupt_state_raises_rather_than_resubmitting(tmp_path):
    """Treating an unreadable file as 'no job' would double-bill a live batch."""
    (tmp_path / "shard0.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not be read"):
        BatchState(str(tmp_path), "shard0")


def test_state_write_is_atomic(tmp_path):
    state = BatchState(str(tmp_path), "shard0")
    state.record(0, "job-0", "oai", 10)
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "shard0.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# engine + config validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine,provider",
    [
        (EngineConfig.OAI, "openai"),
        (EngineConfig.ANTHROPIC, "anthropic"),
        (EngineConfig.ANTHROPIC_AWS, "anthropic"),
        (EngineConfig.VLLM, None),
    ],
)
def test_engine_reports_its_payload_provider(engine, provider):
    assert engine.provider == provider


@pytest.mark.parametrize("engine", [EngineConfig.ANTHROPIC, EngineConfig.ANTHROPIC_AWS])
def test_anthropic_engines_accept_no_sampling_params(engine):
    """The Claude 5 family 400s on temperature/top_p/top_k."""
    assert engine.valid_sampling_params == ["disable_thinking"]


def test_anthropic_config_requires_max_output_tokens():
    with pytest.raises(ValueError, match="max_output_tokens is required"):
        PipeConfig(engine="anthropic_aws", model_name="claude-opus-5", batch=True)


def test_anthropic_config_requires_batch():
    with pytest.raises(ValueError, match="requires batch = true"):
        PipeConfig(
            engine="anthropic_aws", model_name="claude-opus-5", max_output_tokens=4096
        )


def test_local_engine_cannot_use_batch():
    with pytest.raises(ValueError, match="not available for the vllm engine"):
        PipeConfig(engine="vllm", model_name="m", batch=True)


def test_oai_requires_an_api_url():
    with pytest.raises(ValueError, match="api_url is required"):
        PipeConfig(engine="oai", model_name="gpt-5-6", batch=True)


def test_valid_anthropic_batch_config_loads():
    config = PipeConfig(
        engine="anthropic_aws",
        model_name="claude-opus-5",
        batch=True,
        max_output_tokens=16000,
        aws_region="us-east-1",
    )
    assert config.engine.provider == "anthropic"
    assert config.batch
