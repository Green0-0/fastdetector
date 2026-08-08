import json

import pytest

from fastdetector.batch_state import BatchState
from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.toml_config import PipeConfig
from fastdetector.providers.base import BatchResult, order_results
from fastdetector.providers.payloads import build_body, to_anthropic, to_openai


def msgs(*contents) -> list[dict[str, str]]:
    """Build an alternating user/assistant message list from raw contents."""
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": text}
        for i, text in enumerate(contents)
    ]


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------


def test_openai_body_carries_messages_and_params():
    """The OpenAI body passes messages and sampling params through verbatim."""
    body = to_openai(msgs("hi"), {"temperature": 0.7}, "gpt-5-6", 4096)
    assert body["model"] == "gpt-5-6"
    assert body["messages"] == msgs("hi")
    assert body["temperature"] == 0.7
    assert body["max_completion_tokens"] == 4096


def test_anthropic_body_requires_max_tokens():
    """The Messages API rejects a request without max_tokens, so we do too."""
    with pytest.raises(ValueError, match="max_output_tokens is required"):
        to_anthropic(msgs("hi"), {}, "claude-opus-5", None)


def test_anthropic_body_rejects_sampling_params():
    """temperature/top_p/top_k were removed in Claude 5 and return a 400."""
    with pytest.raises(ValueError, match="removed in the Claude 5 family"):
        to_anthropic(msgs("hi"), {"temperature": 0.7}, "claude-opus-5", 4096)


def test_anthropic_body_keeps_thinking_config():
    """disable_thinking maps to a thinking block, which is not a sampling param."""
    body = to_anthropic(msgs("hi"), {"thinking": {"type": "disabled"}}, "claude-opus-5", 4096)
    assert body == {
        "model": "claude-opus-5",
        "max_tokens": 4096,
        "messages": msgs("hi"),
        "thinking": {"type": "disabled"},
    }


def test_anthropic_model_id_is_not_bedrock_prefixed():
    """Claude Platform on AWS takes bare IDs; the prefix belongs to Bedrock."""
    body = to_anthropic(msgs("hi"), {}, "claude-opus-5", 4096)
    assert not body["model"].startswith("anthropic.")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_bodies_reject_empty_message_content(provider):
    """An empty assistant turn is a failed prior generation, never a valid input."""
    with pytest.raises(ValueError, match="empty content"):
        build_body(provider, msgs("ask", "", "again"), {}, "m", 4096)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_bodies_reject_non_alternating_messages(provider):
    """Anthropic requires strict alternation; catch violations before the API does."""
    bad = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    with pytest.raises(ValueError, match="alternate"):
        build_body(provider, bad, {}, "m", 4096)


def test_build_body_rejects_unknown_provider():
    """An unmapped provider is a config error, not a silent passthrough."""
    with pytest.raises(ValueError, match="No payload builder"):
        build_body("gemini", msgs("hi"), {}, "m", 4096)


# --------------------------------------------------------------------------
# result ordering
# --------------------------------------------------------------------------


def test_order_results_restores_caller_order():
    """Batch APIs return results in arbitrary order; index is the identity."""
    found = {2: BatchResult(2, "c", 1, 1), 0: BatchResult(0, "a", 1, 1)}
    ordered = order_results(found, 3)
    assert [r.index for r in ordered] == [0, 1, 2]
    assert [r.text for r in ordered] == ["a", "", "c"]


def test_order_results_pads_missing_indices_as_failures():
    """A dropped request must not shift every later row by one."""
    ordered = order_results({0: BatchResult(0, "a", 1, 1)}, 3)
    assert len(ordered) == 3
    assert ordered[1].failed and "missing" in ordered[1].error


# --------------------------------------------------------------------------
# batch state
# --------------------------------------------------------------------------


def test_state_roundtrips_a_submitted_job(tmp_path):
    """A recorded job is readable by a later process at the same key."""
    BatchState(str(tmp_path), "shard0").record(0, "job-abc", "anthropic_aws", 500)
    record = BatchState(str(tmp_path), "shard0").get(0)
    assert record["job_id"] == "job-abc"
    assert record["n_requests"] == 500


def test_state_keys_turns_separately(tmp_path):
    """Each chat turn is its own batch and its own resume point."""
    state = BatchState(str(tmp_path), "shard0")
    state.record(0, "job-0", "oai", 10)
    state.record(1, "job-1", "oai", 8)
    reloaded = BatchState(str(tmp_path), "shard0")
    assert reloaded.get(0)["job_id"] == "job-0"
    assert reloaded.get(1)["job_id"] == "job-1"


def test_state_is_absent_before_submission(tmp_path):
    """A fresh run has nothing to resume."""
    assert BatchState(str(tmp_path), "shard0").get(0) is None


def test_corrupt_state_raises_rather_than_resubmitting(tmp_path):
    """Treating an unreadable file as 'no job' would double-bill an in-flight batch."""
    state_file = tmp_path / "shard0.json"
    state_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="could not be read"):
        BatchState(str(tmp_path), "shard0")


def test_state_write_is_atomic(tmp_path):
    """No .tmp leftovers, and the file is always complete JSON."""
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
        (EngineConfig.AZURE_OAI, "openai"),
        (EngineConfig.ANTHROPIC, "anthropic"),
        (EngineConfig.ANTHROPIC_AWS, "anthropic"),
        (EngineConfig.VLLM, None),
    ],
)
def test_engine_reports_its_payload_provider(engine, provider):
    """Transport and payload dialect are independent axes."""
    assert engine.provider == provider


@pytest.mark.parametrize("engine", [EngineConfig.ANTHROPIC, EngineConfig.ANTHROPIC_AWS])
def test_anthropic_engines_accept_no_sampling_params(engine):
    """The Claude 5 family 400s on temperature/top_p/top_k."""
    assert engine.valid_sampling_params == ["disable_thinking"]


def test_anthropic_config_requires_max_output_tokens():
    """Catch the missing max_tokens at config load, not mid-run."""
    with pytest.raises(ValueError, match="max_output_tokens is required"):
        PipeConfig(engine="anthropic_aws", model_name="claude-opus-5", batch=True)


def test_anthropic_config_requires_batch():
    """There is no synchronous Anthropic transport in this pipeline."""
    with pytest.raises(ValueError, match="requires batch = true"):
        PipeConfig(
            engine="anthropic_aws", model_name="claude-opus-5", max_output_tokens=4096
        )


def test_local_engine_cannot_use_batch():
    """Offline batching is a hosted-API feature."""
    with pytest.raises(ValueError, match="not available for the vllm engine"):
        PipeConfig(engine="vllm", model_name="m", batch=True)


def test_azure_endpoint_and_version_must_pair():
    """Half an Azure configuration fails at construction, not at request time."""
    with pytest.raises(ValueError, match="must be set together"):
        PipeConfig(
            engine="azure_oai",
            model_name="deployment",
            batch=True,
            azure_endpoint="https://x.openai.azure.com",
        )


def test_valid_anthropic_batch_config_loads():
    """The shipped Claude Platform on AWS shape passes validation."""
    config = PipeConfig(
        engine="anthropic_aws",
        model_name="claude-opus-5",
        batch=True,
        max_output_tokens=16000,
        aws_region="us-east-1",
    )
    assert config.engine.provider == "anthropic"
    assert config.batch
