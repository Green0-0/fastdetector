import json
from types import SimpleNamespace

import pytest

from fastdetector import generator as generator_module
from fastdetector.frontend.toml_config import PipeConfig
from fastdetector.generator import build_dataset
from fastdetector.prompting.prompts import Prompt, PromptSet
from fastdetector.providers.anthropic_batch import AnthropicBatchProvider
from fastdetector.providers import BatchResult, BatchState
from fastdetector.providers.openai_batch import OpenAIBatchProvider


def prompt(turns, use_multiturn=True) -> Prompt:
    """Build a Prompt with the given turns."""
    return Prompt(chat_turns=list(turns), use_multiturn=use_multiturn,
                  examples=[], metadata={})


class FakeProvider:
    """In-memory BatchProvider: records submissions, replays canned outcomes."""

    def __init__(self, name="fake", texts_for=None, polls_before_done=0,
                 shuffle=False, drop_indices=()):
        """Configure the fake.

        Args:
            name: Provider name, as recorded in state.
            texts_for: Callable mapping (index, messages) to response text.
            polls_before_done: How many polls report non-terminal first.
            shuffle: Return results reversed.
            drop_indices: Indices to omit entirely.
        """
        self.name = name
        self.submissions = []
        self.jobs = {}
        self.poll_count = 0
        self._texts_for = texts_for or (lambda i, messages: f"reply:{_last_user(messages)}")
        self._polls_before_done = polls_before_done
        self._shuffle = shuffle
        self._drop = set(drop_indices)

    def submit(self, inputs, generation_params, model_name, max_output_tokens):
        """Record a submission and return a new job ID."""
        job_id = f"job-{len(self.submissions)}"
        self.submissions.append(inputs)
        self.jobs[job_id] = inputs
        return job_id

    def poll(self, job_id):
        """Report non-terminal for the configured number of calls, then done."""
        self.poll_count += 1
        if self.poll_count <= self._polls_before_done:
            return False, "in_progress"
        return True, "completed"

    def fetch(self, job_id, n_requests):
        """Return results, optionally reordered or with indices dropped."""
        found = {
            i: BatchResult(i, self._texts_for(i, messages), 10, 20)
            for i, messages in enumerate(self.jobs[job_id])
            if i not in self._drop
        }
        if self._shuffle:
            found = dict(reversed(list(found.items())))
        return [
            found.get(i, BatchResult(i, "", 0, 0, error="missing from batch output"))
            for i in range(n_requests)
        ]


def _last_user(messages):
    """Extract the final user message content from a conversation."""
    return messages[-1]["content"]


def batch_args(tmp_path, provider, run_key="shard0", **overrides):
    """Build the provider/state/config kwargs for the offline batch path."""
    return {
        "provider": provider,
        "state": BatchState(str(tmp_path), run_key),
        "config": PipeConfig(**{
            "engine": "oai",
            "model_name": "m",
            "api_url": "http://x/v1",
            "batch": True,
            "max_output_tokens": 4096,
            "batch_poll_interval_secs": 0,
            **overrides,
        }),
    }


# --------------------------------------------------------------------------
# End-to-end through build_dataset
# --------------------------------------------------------------------------


def test_offline_batch_produces_the_same_columns_as_the_sync_path(tmp_path):
    provider = FakeProvider()
    columns, prompt_tokens, completion_tokens, failed = build_dataset(
        ["sample one", "sample two"],
        api_url="",
        prompts=PromptSet([prompt(["rewrite {{DOC}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, provider),
    )

    assert set(columns) == {"original", "prompt", "response_0", "final_response"}
    assert columns["response_0"] == ["reply:rewrite sample one", "reply:rewrite sample two"]
    assert (prompt_tokens, completion_tokens, failed) == (20, 40, 0)


def test_results_are_realigned_when_the_provider_returns_them_out_of_order(tmp_path):
    columns, *_ = build_dataset(
        ["alpha", "beta", "gamma"],
        api_url="",
        prompts=PromptSet([prompt(["say {{DOC}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, FakeProvider(shuffle=True)),
    )
    assert columns["response_0"] == ["reply:say alpha", "reply:say beta", "reply:say gamma"]


def test_requests_dropped_by_an_expired_batch_become_aligned_failures(tmp_path):
    columns, _, _, failed = build_dataset(
        ["alpha", "beta", "gamma"],
        api_url="",
        prompts=PromptSet([prompt(["say {{DOC}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, FakeProvider(drop_indices=[1])),
    )
    assert columns["response_0"] == ["reply:say alpha", "", "reply:say gamma"]
    assert columns["original"][2] == "gamma"
    assert failed == 1


def test_one_batch_is_submitted_per_turn(tmp_path):
    provider = FakeProvider()
    build_dataset(
        ["s0", "s1"],
        api_url="",
        prompts=PromptSet([prompt(["a {{DOC}}", "b {{RESP_0}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, provider),
    )
    assert len(provider.submissions) == 2
    assert [len(bodies) for bodies in provider.submissions] == [2, 2]


def test_a_row_that_fails_a_turn_is_not_resubmitted_on_the_next(tmp_path):
    # Key the failure on the sample text, not the within-batch index: after a
    # row is dropped the surviving row shifts down to index 0.
    provider = FakeProvider(
        texts_for=lambda i, messages: "" if "s0" in _last_user(messages)
        else f"reply:{_last_user(messages)}"
    )
    columns, *_ = build_dataset(
        ["s0", "s1"],
        api_url="",
        prompts=PromptSet([prompt(["a {{DOC}}", "b {{RESP_0}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, provider),
    )
    assert [len(bodies) for bodies in provider.submissions] == [2, 1]
    assert columns["final_response"][0] == ""
    assert columns["final_response"][1] != ""


def test_polling_waits_until_the_batch_is_terminal(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(generator_module.time, "sleep", slept.append)
    provider = FakeProvider(polls_before_done=3)
    build_dataset(
        ["s0"],
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, provider),
    )
    assert provider.poll_count == 4
    assert len(slept) == 3


@pytest.mark.parametrize("drop", ["provider", "state", "config"])
def test_partial_batch_arguments_are_rejected(tmp_path, drop):
    """provider/state/config only make sense together."""
    args = batch_args(tmp_path, FakeProvider())
    args[drop] = None
    with pytest.raises(ValueError, match="must be given together"):
        build_dataset(
            ["s0"],
            api_url="",
            prompts=PromptSet([prompt(["go {{DOC}}"])]),
            generation_params={},
            model_name="m",
            **args,
        )


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_the_job_id_is_persisted_at_submission_time(tmp_path):
    build_dataset(
        ["s0"],
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
        **batch_args(tmp_path, FakeProvider()),
    )
    record = BatchState(str(tmp_path), "shard0").get(0)
    assert record["job_id"] == "job-0"
    assert record["complete"] is True


def test_a_rerun_resumes_the_saved_job_instead_of_resubmitting(tmp_path):
    provider = FakeProvider()
    kwargs = dict(
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
    )
    build_dataset(["s0", "s1"], **batch_args(tmp_path, provider), **kwargs)
    assert len(provider.submissions) == 1

    # Same state dir, fresh context - as a requeued SLURM job would see it.
    columns, *_ = build_dataset(
        ["s0", "s1"], **batch_args(tmp_path, provider), **kwargs
    )
    assert len(provider.submissions) == 1, "resume must not create a second job"
    assert columns["response_0"] == ["reply:go s0", "reply:go s1"]


def test_resume_refuses_a_state_file_from_a_different_provider(tmp_path):
    BatchState(str(tmp_path), "shard0").record(0, "job-0", "anthropic_aws", 1)
    with pytest.raises(RuntimeError, match="different provider|configured for"):
        build_dataset(
            ["s0"],
            api_url="",
            prompts=PromptSet([prompt(["go {{DOC}}"])]),
            generation_params={},
            model_name="m",
            **batch_args(tmp_path, FakeProvider(name="oai")),
        )


def test_resume_refuses_when_the_input_count_changed(tmp_path):
    BatchState(str(tmp_path), "shard0").record(0, "job-0", "fake", 99)
    with pytest.raises(RuntimeError, match="covers 99 requests"):
        build_dataset(
            ["s0"],
            api_url="",
            prompts=PromptSet([prompt(["go {{DOC}}"])]),
            generation_params={},
            model_name="m",
            **batch_args(tmp_path, FakeProvider()),
        )


# --------------------------------------------------------------------------
# Vendor wire-format parsing, through fetch() with a stubbed SDK client
# --------------------------------------------------------------------------


def openai_fetch(lines, n_requests=1, status="completed"):
    """Run OpenAIBatchProvider.fetch over canned output-file lines."""
    provider = object.__new__(OpenAIBatchProvider)
    provider.name = "oai"
    job = SimpleNamespace(status=status, output_file_id="f1", error_file_id=None)
    provider.client = SimpleNamespace(
        batches=SimpleNamespace(retrieve=lambda jid: job),
        files=SimpleNamespace(
            content=lambda fid: SimpleNamespace(text="\n".join(lines))
        ),
    )
    return provider.fetch(json.dumps(["b1"]), n_requests)


def anthropic_fetch(entries, n_requests=1):
    """Run AnthropicBatchProvider.fetch over canned result entries."""
    provider = object.__new__(AnthropicBatchProvider)
    provider.name = "anthropic"
    provider.client = SimpleNamespace(
        messages=SimpleNamespace(batches=SimpleNamespace(results=lambda jid: entries))
    )
    return provider.fetch(json.dumps(["b1"]), n_requests)


def succeeded(custom_id, blocks, stop_reason="end_turn", **message_fields):
    """Build a succeeded batch result entry."""
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                stop_reason=stop_reason,
                content=blocks,
                usage=SimpleNamespace(input_tokens=5, output_tokens=6),
                **message_fields,
            ),
        ),
    )


def _line(custom_id, text):
    """Build a successful OpenAI batch output line."""
    return json.dumps({
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }},
    })


def test_fetch_restores_caller_order_from_custom_ids():
    """Batch APIs return results in arbitrary order; custom_id is the identity."""
    results = openai_fetch([_line("req-2", "c"), _line("req-0", "a")], n_requests=3)
    assert [r.index for r in results] == [0, 1, 2]
    assert [r.text for r in results] == ["a", "", "c"]


def test_fetch_pads_indices_the_batch_never_returned():
    """A dropped request must not shift every later row by one."""
    results = openai_fetch([_line("req-0", "a")], n_requests=3)
    assert len(results) == 3
    assert results[0].text == "a"
    assert results[1].failed and "missing" in results[1].error
    assert results[2].failed and "missing" in results[2].error


def test_openai_output_line_is_parsed_into_text_and_usage():
    result = openai_fetch([
        '{"custom_id": "req-0", "response": {"status_code": 200, "body": '
        '{"choices": [{"message": {"content": "hello"}}], '
        '"usage": {"prompt_tokens": 11, "completion_tokens": 22}}}}'
    ])[0]
    assert (result.index, result.text) == (0, "hello")
    assert (result.prompt_tokens, result.completion_tokens) == (11, 22)
    assert not result.failed


@pytest.mark.parametrize(
    "line,reason",
    [
        ('{"custom_id": "req-0", "error": {"message": "boom"}}', "boom"),
        ('{"custom_id": "req-0", "response": {"status_code": 429, "body": {}}}', "429"),
        ('{"custom_id": "req-0", "response": {"status_code": 200, "body": '
         '{"choices": []}}}', "no choices"),
    ],
)
def test_openai_failure_lines_are_reported_with_a_reason(line, reason):
    result = openai_fetch([line])[0]
    assert result.index == 0
    assert result.failed
    assert reason in result.error


def test_anthropic_succeeded_entry_joins_text_blocks():
    result = anthropic_fetch([succeeded("req-0", [
        SimpleNamespace(type="thinking", text="ignored"),
        SimpleNamespace(type="text", text="hello "),
        SimpleNamespace(type="text", text="world"),
    ])])[0]
    assert (result.index, result.text) == (0, "hello world")
    assert (result.prompt_tokens, result.completion_tokens) == (5, 6)


def test_anthropic_refusal_is_a_failure_not_an_empty_success():
    """A decline arrives as a succeeded result with empty content."""
    result = anthropic_fetch([succeeded(
        "req-0", [], stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
    )])[0]
    assert result.failed
    assert "refusal" in result.error and "cyber" in result.error


@pytest.mark.parametrize("outcome", ["expired", "canceled"])
def test_anthropic_non_success_outcomes_are_reported(outcome):
    """expired/canceled carry no error object, so the outcome type is the reason."""
    result = anthropic_fetch([SimpleNamespace(
        custom_id="req-0",
        result=SimpleNamespace(type=outcome, error=None),
    )])[0]
    assert result.index == 0
    assert result.failed
    assert result.error == outcome


def test_anthropic_errored_result_reports_the_nested_error_type():
    """result.error is an ErrorResponse wrapper whose own .type is always "error";
    the useful kind sits one level down at .error.error.type."""
    result = anthropic_fetch([SimpleNamespace(
        custom_id="req-0",
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(
                type="error",
                error=SimpleNamespace(type="rate_limit_error", message="slow down"),
            ),
        ),
    )])[0]
    assert result.failed
    assert result.error == "rate_limit_error"
