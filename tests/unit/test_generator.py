"""Message construction and the multi-turn dataset builder.

``batch_generate`` is exercised against a real OpenAI-compatible server running
on localhost (see the ``fake_openai_server`` fixture), so the actual client,
serialisation, and concurrency paths run — only the model is fake.
"""

import pytest

from fastdetector import generator as generator_module
from fastdetector.generator import _build_messages, batch_generate, build_dataset
from fastdetector.prompting.prompts import Prompt, PromptSet


def prompt(turns, use_multiturn=True, examples=None, metadata=None) -> Prompt:
    """Build a Prompt with the given turns."""
    return Prompt(
        chat_turns=list(turns),
        use_multiturn=use_multiturn,
        examples=list(examples or []),
        metadata=dict(metadata or {}),
    )


# --------------------------------------------------------------------------
# _build_messages
# --------------------------------------------------------------------------


def test_first_turn_is_a_single_user_message():
    messages = _build_messages(prompt(["turn zero"]), turn_index=0, responses=[])
    assert messages == [{"role": "user", "content": "turn zero"}]


def test_examples_are_prepended_as_user_assistant_pairs():
    messages = _build_messages(
        prompt(["ask"], examples=[("ex user", "ex assistant")]),
        turn_index=0,
        responses=[],
    )
    assert messages == [
        {"role": "user", "content": "ex user"},
        {"role": "assistant", "content": "ex assistant"},
        {"role": "user", "content": "ask"},
    ]


def test_multiturn_replays_the_whole_conversation():
    messages = _build_messages(
        prompt(["first", "second"]), turn_index=1, responses=["answer0"]
    )
    assert messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer0"},
        {"role": "user", "content": "second"},
    ]


def test_single_turn_mode_sends_only_the_latest_message():
    messages = _build_messages(
        prompt(["first", "second"], use_multiturn=False),
        turn_index=1,
        responses=["answer0"],
    )
    assert messages == [{"role": "user", "content": "second"}]


def test_single_turn_mode_drops_the_examples_after_the_first_turn():
    # With use_multiturn=False each later turn is a brand new request whose
    # only context is the {{RESP_N}} substitutions.
    messages = _build_messages(
        prompt(["first", "second"], use_multiturn=False, examples=[("u", "a")]),
        turn_index=1,
        responses=["answer0"],
    )
    assert all(m["role"] == "user" for m in messages)
    assert len(messages) == 1


def test_single_turn_mode_still_sends_examples_on_the_first_turn():
    messages = _build_messages(
        prompt(["first"], use_multiturn=False, examples=[("u", "a")]),
        turn_index=0,
        responses=[],
    )
    assert len(messages) == 3


def test_resp_placeholders_are_substituted():
    messages = _build_messages(
        prompt(["first", "expand on {{RESP_0}} please"], use_multiturn=False),
        turn_index=1,
        responses=["THE ANSWER"],
    )
    assert messages[0]["content"] == "expand on THE ANSWER please"


def test_multiple_resp_placeholders_are_substituted():
    messages = _build_messages(
        prompt(["a", "b", "{{RESP_0}} then {{RESP_1}}"], use_multiturn=False),
        turn_index=2,
        responses=["ZERO", "ONE"],
    )
    assert messages[0]["content"] == "ZERO then ONE"


def test_only_earlier_responses_are_substituted():
    # {{RESP_1}} inside turn 1 refers to that turn's own (not yet produced)
    # answer, so it must be left alone rather than raising IndexError.
    messages = _build_messages(
        prompt(["a", "self {{RESP_1}}"], use_multiturn=False),
        turn_index=1,
        responses=["ZERO"],
    )
    assert messages[0]["content"] == "self {{RESP_1}}"


# --------------------------------------------------------------------------
# build_dataset
# --------------------------------------------------------------------------


@pytest.fixture
def recording_batch_generate(monkeypatch):
    """Replace batch_generate with a deterministic echo and record its calls."""
    calls = []

    def fake(api_url, inputs, generation_params, api_key="EMPTY", model_name=""):
        calls.append(
            {
                "api_url": api_url,
                "inputs": inputs,
                "generation_params": generation_params,
                "api_key": api_key,
                "model_name": model_name,
            }
        )
        texts = [f"reply{len(calls) - 1}:{msgs[-1]['content']}" for msgs in inputs]
        return texts, 10 * len(inputs), 20 * len(inputs), 0

    monkeypatch.setattr(generator_module, "batch_generate", fake)
    return calls


def test_build_dataset_single_turn_columns(recording_batch_generate):
    prompts = PromptSet([prompt(["rewrite {{DOC}}"])])
    columns, prompt_tokens, completion_tokens, failed = build_dataset(
        ["sample one", "sample two"], "http://x/v1", prompts, {}
    )

    assert set(columns) == {"original", "prompt", "response_0", "final_response"}
    assert columns["original"] == ["sample one", "sample two"]
    assert columns["response_0"] == [
        "reply0:rewrite sample one",
        "reply0:rewrite sample two",
    ]
    assert columns["final_response"] == columns["response_0"]
    assert (prompt_tokens, completion_tokens, failed) == (20, 40, 0)


def test_build_dataset_every_column_has_one_row_per_sample(recording_batch_generate):
    prompts = PromptSet([prompt(["a {{DOC}}"]), prompt(["b {{DOC}}", "c"])])
    columns, *_ = build_dataset(["s0", "s1", "s2"], "http://x/v1", prompts, {})
    assert {len(v) for v in columns.values()} == {3}


def test_build_dataset_fills_missing_turns_with_empty_strings(recording_batch_generate):
    # Prompt 0 has one turn, prompt 1 has two; row 0 has no response_1.
    prompts = PromptSet([prompt(["one {{DOC}}"]), prompt(["two {{DOC}}", "more"])])
    columns, *_ = build_dataset(["s0", "s1"], "http://x/v1", prompts, {})
    assert columns["response_1"][0] == ""
    assert columns["response_1"][1] != ""


def test_build_dataset_final_response_is_each_rows_last_answer(
    recording_batch_generate,
):
    prompts = PromptSet([prompt(["one {{DOC}}"]), prompt(["two {{DOC}}", "more"])])
    columns, *_ = build_dataset(["s0", "s1"], "http://x/v1", prompts, {})
    assert columns["final_response"][0] == columns["response_0"][0]
    assert columns["final_response"][1] == columns["response_1"][1]


def test_build_dataset_batches_once_per_turn(recording_batch_generate):
    prompts = PromptSet([prompt(["a {{DOC}}", "b", "c"])])
    build_dataset(["s0", "s1"], "http://x/v1", prompts, {})
    assert len(recording_batch_generate) == 3
    assert [len(call["inputs"]) for call in recording_batch_generate] == [2, 2, 2]


def test_build_dataset_only_sends_rows_that_have_the_turn(recording_batch_generate):
    prompts = PromptSet([prompt(["one {{DOC}}"]), prompt(["two {{DOC}}", "more"])])
    build_dataset(["s0", "s1"], "http://x/v1", prompts, {})
    assert [len(call["inputs"]) for call in recording_batch_generate] == [2, 1]


def test_build_dataset_feeds_earlier_responses_into_later_turns(
    recording_batch_generate,
):
    prompts = PromptSet([prompt(["first {{DOC}}", "use {{RESP_0}}"])])
    build_dataset(["s0"], "http://x/v1", prompts, {})
    second_turn_message = recording_batch_generate[1]["inputs"][0][-1]["content"]
    assert second_turn_message == "use reply0:first s0"


def test_build_dataset_accumulates_usage_across_turns(recording_batch_generate):
    prompts = PromptSet([prompt(["a {{DOC}}", "b"])])
    _, prompt_tokens, completion_tokens, _ = build_dataset(
        ["s0", "s1"], "http://x/v1", prompts, {}
    )
    assert prompt_tokens == 40
    assert completion_tokens == 80


def test_build_dataset_accumulates_failures(monkeypatch):
    monkeypatch.setattr(
        generator_module,
        "batch_generate",
        lambda *a, **k: ([""] * len(a[1]), 0, 0, len(a[1])),
    )
    prompts = PromptSet([prompt(["a {{DOC}}", "b"])])
    *_, failed = build_dataset(["s0", "s1"], "http://x/v1", prompts, {})
    assert failed == 4


def test_build_dataset_forwards_credentials_and_params(recording_batch_generate):
    prompts = PromptSet([prompt(["a {{DOC}}"])])
    build_dataset(
        ["s0"],
        "http://x/v1",
        prompts,
        {"temperature": 0.5},
        api_key="secret",
        model_name="some/model",
    )
    call = recording_batch_generate[0]
    assert call["api_key"] == "secret"
    assert call["model_name"] == "some/model"
    assert call["generation_params"] == {"temperature": 0.5}


def test_build_dataset_with_no_samples_short_circuits(recording_batch_generate):
    prompts = PromptSet([prompt(["a {{DOC}}"])])
    columns, prompt_tokens, completion_tokens, failed = build_dataset(
        [], "http://x/v1", prompts, {}
    )
    assert columns == {"original": [], "prompt": [], "final_response": []}
    assert (prompt_tokens, completion_tokens, failed) == (0, 0, 0)
    assert recording_batch_generate == []


def test_build_dataset_records_the_prompt_template_per_row(recording_batch_generate):
    prompts = PromptSet([prompt(["rewrite {{DOC}}"], metadata={"PROMPT_TYPE": "rw"})])
    columns, *_ = build_dataset(["s0"], "http://x/v1", prompts, {})
    assert columns["prompt"][0]["metadata"] == {"PROMPT_TYPE": "rw"}
    assert columns["prompt"][0]["chat_turns"] == ["rewrite {{DOC}}"]


# --------------------------------------------------------------------------
# batch_generate against a live localhost endpoint
# --------------------------------------------------------------------------


def test_batch_generate_returns_responses_in_request_order(fake_openai_server):
    inputs = [[{"role": "user", "content": f"msg{i}"}] for i in range(6)]
    texts, prompt_tokens, completion_tokens, failed = batch_generate(
        fake_openai_server.url, inputs, {}, model_name="fake-model"
    )
    assert texts == [f"echo:msg{i}" for i in range(6)]
    assert prompt_tokens == 18
    assert completion_tokens == 30
    assert failed == 0


def test_batch_generate_sends_the_model_and_sampling_params(fake_openai_server):
    batch_generate(
        fake_openai_server.url,
        [[{"role": "user", "content": "hi"}]],
        {"temperature": 0.25, "extra_body": {"top_k": 20}},
        model_name="some/model",
    )
    payload = fake_openai_server.received[0]
    assert payload["model"] == "some/model"
    assert payload["temperature"] == 0.25
    assert payload["top_k"] == 20


def test_batch_generate_counts_content_filtered_responses_as_failures(
    fake_openai_server,
):
    def no_choices(payload):
        return 200, {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [],
        }

    fake_openai_server.set_responder(no_choices)
    texts, _, _, failed = batch_generate(
        fake_openai_server.url, [[{"role": "user", "content": "hi"}]] * 3, {}
    )
    assert texts == ["", "", ""]
    assert failed == 3


def test_batch_generate_keeps_failed_rows_aligned(fake_openai_server):
    def fail_the_middle_one(payload):
        content = payload["messages"][-1]["content"]
        if content == "msg1":
            return 400, {"error": {"message": "nope", "type": "invalid_request_error"}}
        return 200, {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"ok:{content}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    fake_openai_server.set_responder(fail_the_middle_one)
    inputs = [[{"role": "user", "content": f"msg{i}"}] for i in range(3)]
    texts, _, _, failed = batch_generate(fake_openai_server.url, inputs, {})
    assert texts == ["ok:msg0", "", "ok:msg2"]
    assert failed == 1


def test_batch_generate_treats_a_null_content_as_an_empty_string(fake_openai_server):
    def null_content(payload):
        return 200, {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
        }

    fake_openai_server.set_responder(null_content)
    texts, prompt_tokens, _, failed = batch_generate(
        fake_openai_server.url, [[{"role": "user", "content": "hi"}]], {}
    )
    assert texts == [""]
    # Usage was reported, so this is a truncated answer rather than a failure.
    assert prompt_tokens == 7
    assert failed == 0


def test_batch_generate_with_no_inputs_does_not_call_the_server(fake_openai_server):
    assert batch_generate(fake_openai_server.url, [], {}) == ([], 0, 0, 0)
    assert fake_openai_server.received == []


def test_build_dataset_end_to_end_against_the_fake_server(fake_openai_server):
    prompts = PromptSet(
        [prompt(["rewrite {{DOC}}", "polish {{RESP_0}}"], use_multiturn=False)]
    )
    columns, prompt_tokens, completion_tokens, failed = build_dataset(
        ["document one", "document two"],
        fake_openai_server.url,
        prompts,
        {},
        model_name="fake-model",
    )
    assert columns["response_0"] == [
        "echo:rewrite document one",
        "echo:rewrite document two",
    ]
    assert columns["final_response"] == [
        "echo:polish echo:rewrite document one",
        "echo:polish echo:rewrite document two",
    ]
    assert failed == 0
    assert prompt_tokens > 0 and completion_tokens > 0
