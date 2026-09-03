import pytest

from fastdetector import generator as generator_module
from fastdetector.generation_checkpoint import GenerationCheckpoint
from fastdetector.generator import _RequestResult, build_dataset
from fastdetector.prompting.prompts import Prompt, PromptSet


FINGERPRINT = {"model": "m", "inputs": "digest"}


def open_checkpoint(tmp_path, run_key="shard_0", fingerprint=FINGERPRINT):
    checkpoint = GenerationCheckpoint(
        str(tmp_path), run_key, flush_records=1, flush_seconds=3600
    )
    checkpoint.prepare(fingerprint)
    return checkpoint


def prompt(turns):
    return Prompt(list(turns), True, [], {})


def test_resume_sends_only_pending_rows_and_restores_original_order(tmp_path, monkeypatch):
    attempted = []
    fail_middle = True

    async def fake_send(client, semaphore, messages, generation_params, model_name=""):
        nonlocal fail_middle
        content = messages[-1]["content"]
        attempted.append(content)
        if content == "row one" and fail_middle:
            return _RequestResult("", 7, 0, "temporary failure")
        return _RequestResult(f"answer:{content}", 2, 3)

    monkeypatch.setattr(generator_module, "_send_request", fake_send)
    samples = [f"row {name}" for name in ("zero", "one", "two")]

    first = open_checkpoint(tmp_path)
    try:
        columns, _, _, failed = build_dataset(
            samples, "http://unused/v1", PromptSet([prompt(["{{DOC}}"])]), {},
            checkpoint=first,
        )
        assert columns["final_response"] == ["answer:row zero", "", "answer:row two"]
        assert failed == 1
    finally:
        first.close()

    attempted.clear()
    fail_middle = False
    resumed = open_checkpoint(tmp_path)
    try:
        columns, prompt_tokens, completion_tokens, failed = build_dataset(
            samples, "http://unused/v1", PromptSet([prompt(["{{DOC}}"])]), {},
            checkpoint=resumed,
        )
        assert attempted == ["row one"]
        assert columns["final_response"] == [
            "answer:row zero", "answer:row one", "answer:row two"
        ]
        assert (prompt_tokens, completion_tokens, failed) == (6, 9, 0)
    finally:
        resumed.close()


def test_a_torn_final_line_is_removed_before_new_records_are_appended(tmp_path):
    checkpoint = open_checkpoint(tmp_path)
    log = checkpoint.turn(0)
    log.append(0, "saved", 1, 2)
    checkpoint.close()
    path = tmp_path / "shard_0" / "turn_0.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"version":1,"turn":0,"row_id":1')

    resumed = open_checkpoint(tmp_path)
    try:
        resumed_log = resumed.turn(0)
        assert set(resumed_log.records) == {0}
        resumed_log.append(1, "recovered", 3, 4)
    finally:
        resumed.close()

    final = open_checkpoint(tmp_path)
    try:
        assert set(final.turn(0).records) == {0, 1}
    finally:
        final.close()


def test_append_flushes_the_python_buffer_immediately(tmp_path):
    checkpoint = GenerationCheckpoint(
        str(tmp_path), "buffered", flush_records=100, flush_seconds=3600
    )
    checkpoint.prepare(FINGERPRINT)
    try:
        checkpoint.turn(0).append(0, "saved", 1, 2)
        path = tmp_path / "buffered" / "turn_0.jsonl"
        assert '"text": "saved"' in path.read_text()
    finally:
        checkpoint.close()


def test_corruption_before_the_final_line_is_not_silently_skipped(tmp_path):
    checkpoint = open_checkpoint(tmp_path)
    checkpoint.close()
    path = tmp_path / "shard_0" / "turn_0.jsonl"
    path.write_text("not json\nnot json either\n", encoding="utf-8")

    resumed = open_checkpoint(tmp_path)
    try:
        with pytest.raises(RuntimeError, match=r"turn_0.jsonl:1"):
            resumed.turn(0)
    finally:
        resumed.close()


def test_fingerprint_mismatch_archives_instead_of_mixing_results(tmp_path):
    first = open_checkpoint(tmp_path)
    first.turn(0).append(0, "old answer", 1, 1)
    first.close()

    changed = open_checkpoint(tmp_path, fingerprint={"model": "different"})
    try:
        assert changed.turn(0).records == {}
        archives = list(tmp_path.glob("shard_0.stale-*"))
        assert len(archives) == 1
        assert (archives[0] / "turn_0.jsonl").exists()
    finally:
        changed.close()


def test_only_one_process_can_use_a_checkpoint(tmp_path):
    first = open_checkpoint(tmp_path)
    second = GenerationCheckpoint(str(tmp_path), "shard_0")
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            second.prepare(FINGERPRINT)
    finally:
        first.close()

    second.prepare(FINGERPRINT)
    second.close()


def test_retire_removes_results_only_after_the_caller_commits(tmp_path):
    checkpoint = open_checkpoint(tmp_path)
    checkpoint.turn(0).append(0, "answer", 1, 1)
    assert checkpoint.path.exists()
    checkpoint.retire()
    assert not checkpoint.path.exists()
    checkpoint.close()


def test_multiturn_resume_uses_absolute_row_ids_when_eligibility_changes(
    tmp_path, monkeypatch
):
    calls = []
    first_run = True

    def fake_generate(api_url, inputs, generation_params, api_key="EMPTY",
                      model_name="", row_ids=None, turn_log=None):
        calls.append(list(row_ids))
        texts = []
        for row_id in row_ids:
            if first_run and row_id == 0 and len(calls) == 1:
                texts.append("")
            else:
                texts.append(f"turn{turn_log.turn_index}:row{row_id}")
        for row_id, text in zip(row_ids, texts):
            if text:
                turn_log.append(row_id, text, 1, 1)
        return texts, sum(bool(text) for text in texts), sum(bool(text) for text in texts), texts.count("")

    monkeypatch.setattr(generator_module, "batch_generate", fake_generate)
    checkpoint = open_checkpoint(tmp_path)
    try:
        build_dataset(
            ["s0", "s1"], "http://unused/v1",
            PromptSet([prompt(["first {{DOC}}", "second"])]), {},
            checkpoint=checkpoint,
        )
    finally:
        checkpoint.close()
    assert calls == [[0, 1], [1]]

    calls.clear()
    first_run = False
    resumed = open_checkpoint(tmp_path)
    try:
        columns, *_ = build_dataset(
            ["s0", "s1"], "http://unused/v1",
            PromptSet([prompt(["first {{DOC}}", "second"])]), {},
            checkpoint=resumed,
        )
        assert calls == [[0], [0]]
        assert columns["response_1"] == ["turn1:row0", "turn1:row1"]
    finally:
        resumed.close()


def test_failure_guard_aborts_before_an_empty_shard_can_be_returned(tmp_path, monkeypatch):
    def all_fail(api_url, inputs, generation_params, api_key="EMPTY",
                 model_name="", row_ids=None, turn_log=None):
        return [""] * len(inputs), 0, 0, len(inputs)

    monkeypatch.setattr(generator_module, "batch_generate", all_fail)
    checkpoint = open_checkpoint(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="appears unhealthy"):
            build_dataset(
                ["a", "b", "c", "d"], "http://unused/v1",
                PromptSet([prompt(["rewrite {{DOC}}"])]), {},
                checkpoint=checkpoint, max_failure_rate=0.25,
            )
        assert checkpoint.path.exists()
    finally:
        checkpoint.close()


def test_failure_guard_uses_all_active_rows_after_resume(tmp_path, monkeypatch):
    checkpoint = open_checkpoint(tmp_path)
    for row_id in range(3):
        checkpoint.turn(0).append(row_id, f"saved {row_id}", 1, 1)
    checkpoint.close()

    def fail_remaining(api_url, inputs, generation_params, api_key="EMPTY",
                       model_name="", row_ids=None, turn_log=None):
        return [""] * len(inputs), 0, 0, len(inputs)

    monkeypatch.setattr(generator_module, "batch_generate", fail_remaining)
    resumed = open_checkpoint(tmp_path)
    try:
        columns, _, _, failed = build_dataset(
            ["a", "b", "c", "d"], "http://unused/v1",
            PromptSet([prompt(["rewrite {{DOC}}"])]), {},
            checkpoint=resumed, max_failure_rate=0.25,
        )
        assert columns["final_response"] == ["saved 0", "saved 1", "saved 2", ""]
        assert failed == 1
    finally:
        resumed.close()
