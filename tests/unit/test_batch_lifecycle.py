"""Credential-free coverage of the offline-batch lifecycle.

The provider classes are thin wrappers over vendor SDK calls, but everything
around them - submit/poll/fetch sequencing, resume after a killed process,
index realignment, failure propagation - is ours and is where the expensive
mistakes live (a lost job ID means paying for the same batch twice).

These tests drive that logic through a fake provider that implements the
BatchProvider protocol in memory, so the whole path is exercised with no API
key, no network, and no billing. The vendor response *shapes* are covered
separately by the parser tests at the bottom, using synthetic payloads in the
documented wire format.
"""
from types import SimpleNamespace

import pytest

from fastdetector import generator as generator_module
from fastdetector.batch_state import BatchState
from fastdetector.generator import BatchContext, build_dataset
from fastdetector.prompting.prompts import Prompt, PromptSet
from fastdetector.providers.base import BatchResult, order_results


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
            texts_for: Callable mapping (index, body) to response text, or None
                for a deterministic echo.
            polls_before_done: How many polls report non-terminal first.
            shuffle: Return results reversed, mimicking the arbitrary ordering
                batch APIs actually give.
            drop_indices: Indices to omit entirely, mimicking an expired batch.
        """
        self.name = name
        self.submissions = []
        self.jobs = {}
        self.poll_count = 0
        self._texts_for = texts_for or (lambda i, body: f"reply:{_last_user(body)}")
        self._polls_before_done = polls_before_done
        self._shuffle = shuffle
        self._drop = set(drop_indices)

    def submit(self, bodies):
        """Record a submission and return a new job ID."""
        job_id = f"job-{len(self.submissions)}"
        self.submissions.append(bodies)
        self.jobs[job_id] = bodies
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
            i: BatchResult(i, self._texts_for(i, body), 10, 20)
            for i, body in enumerate(self.jobs[job_id])
            if i not in self._drop
        }
        if self._shuffle:
            found = dict(reversed(list(found.items())))
        return order_results(found, n_requests)


def _last_user(body):
    """Extract the final user message content from either payload dialect."""
    return body["messages"][-1]["content"]


def context(tmp_path, provider, provider_name="openai", max_output_tokens=4096,
            run_key="shard0"):
    """Build a BatchContext wired to *provider* and a temp state dir."""
    return BatchContext(
        provider=provider,
        provider_name=provider_name,
        state=BatchState(str(tmp_path), run_key),
        max_output_tokens=max_output_tokens,
        poll_interval_secs=0,
    )


# --------------------------------------------------------------------------
# End-to-end through build_dataset
# --------------------------------------------------------------------------


def test_offline_batch_produces_the_same_columns_as_the_sync_path(tmp_path):
    """The batch transport is a drop-in: same contract, same output shape."""
    provider = FakeProvider()
    columns, prompt_tokens, completion_tokens, failed = build_dataset(
        ["sample one", "sample two"],
        api_url="",
        prompts=PromptSet([prompt(["rewrite {{DOC}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, provider),
    )

    assert set(columns) == {"original", "prompt", "response_0", "final_response"}
    assert columns["response_0"] == ["reply:rewrite sample one", "reply:rewrite sample two"]
    assert (prompt_tokens, completion_tokens, failed) == (20, 40, 0)


def test_results_are_realigned_when_the_provider_returns_them_out_of_order(tmp_path):
    """Batch APIs guarantee no ordering; rows must key on index, not position."""
    columns, *_ = build_dataset(
        ["alpha", "beta", "gamma"],
        api_url="",
        prompts=PromptSet([prompt(["say {{DOC}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, FakeProvider(shuffle=True)),
    )
    assert columns["response_0"] == ["reply:say alpha", "reply:say beta", "reply:say gamma"]


def test_requests_dropped_by_an_expired_batch_become_aligned_failures(tmp_path):
    """A partial batch must not shift later rows onto the wrong source text."""
    columns, _, _, failed = build_dataset(
        ["alpha", "beta", "gamma"],
        api_url="",
        prompts=PromptSet([prompt(["say {{DOC}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, FakeProvider(drop_indices=[1])),
    )
    assert columns["response_0"] == ["reply:say alpha", "", "reply:say gamma"]
    assert columns["original"][2] == "gamma"
    assert failed == 1


def test_one_batch_is_submitted_per_turn(tmp_path):
    """Turns are strictly sequential: turn N+1 embeds turn N's text."""
    provider = FakeProvider()
    build_dataset(
        ["s0", "s1"],
        api_url="",
        prompts=PromptSet([prompt(["a {{DOC}}", "b {{RESP_0}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, provider),
    )
    assert len(provider.submissions) == 2
    assert [len(bodies) for bodies in provider.submissions] == [2, 2]


def test_a_row_that_fails_a_turn_is_not_resubmitted_on_the_next(tmp_path):
    """Under batch pricing, replaying a dead row is paid-for garbage."""
    # Key the failure on the sample text, not the within-batch index: after a
    # row is dropped the surviving row shifts down to index 0.
    provider = FakeProvider(
        texts_for=lambda i, body: "" if "s0" in _last_user(body)
        else f"reply:{_last_user(body)}"
    )
    columns, *_ = build_dataset(
        ["s0", "s1"],
        api_url="",
        prompts=PromptSet([prompt(["a {{DOC}}", "b {{RESP_0}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, provider),
    )
    assert [len(bodies) for bodies in provider.submissions] == [2, 1]
    assert columns["final_response"][0] == ""
    assert columns["final_response"][1] != ""


def test_polling_waits_until_the_batch_is_terminal(tmp_path, monkeypatch):
    """A 24h window means most of the run is spent here."""
    slept = []
    monkeypatch.setattr(generator_module.time, "sleep", slept.append)
    provider = FakeProvider(polls_before_done=3)
    build_dataset(
        ["s0"],
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, provider),
    )
    assert provider.poll_count == 4
    assert len(slept) == 3


# --------------------------------------------------------------------------
# Resume - the reason state exists at all
# --------------------------------------------------------------------------


def test_the_job_id_is_persisted_at_submission_time(tmp_path):
    """It must be durable before polling starts, not after results arrive."""
    build_dataset(
        ["s0"],
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
        batch_ctx=context(tmp_path, FakeProvider()),
    )
    record = BatchState(str(tmp_path), "shard0").get(0)
    assert record["job_id"] == "job-0"
    assert record["complete"] is True


def test_a_rerun_resumes_the_saved_job_instead_of_resubmitting(tmp_path):
    """Resubmitting an in-flight batch pays for the same work twice."""
    provider = FakeProvider()
    kwargs = dict(
        api_url="",
        prompts=PromptSet([prompt(["go {{DOC}}"])]),
        generation_params={},
        model_name="m",
    )
    build_dataset(["s0", "s1"], batch_ctx=context(tmp_path, provider), **kwargs)
    assert len(provider.submissions) == 1

    # Same state dir, fresh context - as a requeued SLURM job would see it.
    columns, *_ = build_dataset(
        ["s0", "s1"], batch_ctx=context(tmp_path, provider), **kwargs
    )
    assert len(provider.submissions) == 1, "resume must not create a second job"
    assert columns["response_0"] == ["reply:go s0", "reply:go s1"]


def test_resume_refuses_a_state_file_from_a_different_provider(tmp_path):
    """Job IDs are provider-scoped; crossing them would fetch nonsense."""
    state = BatchState(str(tmp_path), "shard0")
    state.record(0, "job-0", "anthropic_aws", 1)
    with pytest.raises(RuntimeError, match="different provider|configured for"):
        build_dataset(
            ["s0"],
            api_url="",
            prompts=PromptSet([prompt(["go {{DOC}}"])]),
            generation_params={},
            model_name="m",
            batch_ctx=context(tmp_path, FakeProvider(name="oai")),
        )


def test_resume_refuses_when_the_input_count_changed(tmp_path):
    """A changed shard or prompt file invalidates the saved index mapping."""
    state = BatchState(str(tmp_path), "shard0")
    state.record(0, "job-0", "fake", 99)
    with pytest.raises(RuntimeError, match="covers 99 requests"):
        build_dataset(
            ["s0"],
            api_url="",
            prompts=PromptSet([prompt(["go {{DOC}}"])]),
            generation_params={},
            model_name="m",
            batch_ctx=context(tmp_path, FakeProvider()),
        )


# --------------------------------------------------------------------------
# Vendor wire-format parsing (synthetic payloads, no SDK calls)
# --------------------------------------------------------------------------


def test_openai_output_line_is_parsed_into_text_and_usage():
    """Happy-path shape of an OpenAI batch output JSONL line."""
    from fastdetector.providers.openai_batch import _parse_line

    result = _parse_line(
        '{"custom_id": "req-7", "response": {"status_code": 200, "body": '
        '{"choices": [{"message": {"content": "hello"}}], '
        '"usage": {"prompt_tokens": 11, "completion_tokens": 22}}}}'
    )
    assert (result.index, result.text) == (7, "hello")
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
    """Every failure mode keeps its index and explains itself."""
    from fastdetector.providers.openai_batch import _parse_line

    result = _parse_line(line)
    assert result.index == 0
    assert result.failed
    assert reason in result.error


def test_anthropic_succeeded_entry_joins_text_blocks():
    """Content is a block list; only text blocks contribute."""
    from fastdetector.providers.anthropic_batch import _parse_entry

    entry = SimpleNamespace(
        custom_id="req-3",
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                stop_reason="end_turn",
                content=[
                    SimpleNamespace(type="thinking", text="ignored"),
                    SimpleNamespace(type="text", text="hello "),
                    SimpleNamespace(type="text", text="world"),
                ],
                usage=SimpleNamespace(input_tokens=5, output_tokens=6),
            ),
        ),
    )
    result = _parse_entry(entry)
    assert (result.index, result.text) == (3, "hello world")
    assert (result.prompt_tokens, result.completion_tokens) == (5, 6)


def test_anthropic_refusal_is_a_failure_not_an_empty_success():
    """Claude 5 declines arrive as a *succeeded* result with empty content."""
    from fastdetector.providers.anthropic_batch import _parse_entry

    entry = SimpleNamespace(
        custom_id="req-1",
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                stop_reason="refusal",
                stop_details=SimpleNamespace(category="cyber"),
                content=[],
                usage=SimpleNamespace(input_tokens=5, output_tokens=0),
            ),
        ),
    )
    result = _parse_entry(entry)
    assert result.failed
    assert "refusal" in result.error and "cyber" in result.error


@pytest.mark.parametrize("outcome", ["errored", "expired", "canceled"])
def test_anthropic_non_success_outcomes_are_reported(outcome):
    """expired means the 24h window lapsed; those were never billed."""
    from fastdetector.providers.anthropic_batch import _parse_entry

    entry = SimpleNamespace(
        custom_id="req-2",
        result=SimpleNamespace(type=outcome, error=None),
    )
    result = _parse_entry(entry)
    assert result.index == 2
    assert result.failed
