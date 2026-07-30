import json

import pytest

from fastdetector.prompting.prompts import Prompt, PromptSet, load_prompts


def make_prompts(n: int, turns: int = 1) -> list[Prompt]:
    """Build ``n`` identifiable prompts, each with ``turns`` chat turns."""
    return [
        Prompt(
            chat_turns=[f"p{i}t{t}:{{{{DOC}}}}" for t in range(turns)],
            use_multiturn=True,
            metadata={"PROMPT_TYPE": f"type{i}"},
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Prompt dataclass
# --------------------------------------------------------------------------


def test_prompt_defaults_are_not_shared_between_instances():
    a = Prompt(chat_turns=["x"], use_multiturn=True)
    b = Prompt(chat_turns=["y"], use_multiturn=True)
    a.metadata["k"] = 1
    a.examples.append(("u", "a"))
    assert b.metadata == {}
    assert b.examples == []


# --------------------------------------------------------------------------
# Cursors
# --------------------------------------------------------------------------


def test_next_train_advances_in_order():
    prompt_set = PromptSet(make_prompts(3))
    first = prompt_set.next_train(2)
    second = prompt_set.next_train(2)
    assert [p.chat_turns[0] for p in first] == ["p0t0:{{DOC}}", "p1t0:{{DOC}}"]
    assert [p.chat_turns[0] for p in second] == ["p2t0:{{DOC}}", "p0t0:{{DOC}}"]


def test_next_train_wraps_within_a_single_call():
    prompt_set = PromptSet(make_prompts(2))
    got = prompt_set.next_train(5)
    assert [p.chat_turns[0][:2] for p in got] == ["p0", "p1", "p0", "p1", "p0"]


def test_next_train_zero_returns_empty_and_does_not_move_the_cursor():
    prompt_set = PromptSet(make_prompts(2))
    assert prompt_set.next_train(0) == []
    assert prompt_set.next_train(1)[0].chat_turns[0].startswith("p0")


def test_next_train_on_empty_set_raises():
    with pytest.raises(RuntimeError, match="training set is empty"):
        PromptSet([]).next_train(1)


def test_next_test_without_a_split_raises():
    with pytest.raises(RuntimeError, match="testing set is empty"):
        PromptSet(make_prompts(3)).next_test(1)


def test_reset_rewinds_both_cursors():
    prompt_set = PromptSet(make_prompts(4))
    prompt_set.generate_test_split(0.5)
    prompt_set.next_train(1)
    prompt_set.next_test(1)
    prompt_set.reset()
    assert prompt_set.next_train(1)[0].chat_turns[0].startswith("p0")
    assert prompt_set.next_test(1)[0].chat_turns[0].startswith("p2")


# --------------------------------------------------------------------------
# Test splits
# --------------------------------------------------------------------------


def test_generate_test_split_partitions_without_losing_prompts():
    prompt_set = PromptSet(make_prompts(10))
    prompt_set.generate_test_split(0.3)
    assert len(prompt_set.get_train()) == 7
    assert len(prompt_set.get_test()) == 3


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_generate_test_split_rejects_out_of_range_fractions(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        PromptSet(make_prompts(10)).generate_test_split(fraction)


def test_generate_test_split_refuses_to_empty_the_training_set():
    # int() flooring would otherwise move the only prompt into the test set and
    # blow up later inside next_train().
    with pytest.raises(ValueError, match="would leave the training set"):
        PromptSet(make_prompts(1)).generate_test_split(0.5)


def test_resplitting_merges_the_previous_test_set_back_first():
    prompt_set = PromptSet(make_prompts(10))
    prompt_set.generate_test_split(0.5)
    prompt_set.generate_test_split(0.2)
    assert len(prompt_set.get_train()) == 8
    assert len(prompt_set.get_test()) == 2
    names = {p.chat_turns[0] for p in prompt_set.get_train() + prompt_set.get_test()}
    assert len(names) == 10


def test_clear_test_set_restores_everything_to_train():
    prompt_set = PromptSet(make_prompts(6))
    prompt_set.generate_test_split(0.5)
    prompt_set.clear_test_set()
    assert len(prompt_set.get_train()) == 6
    assert prompt_set.get_test() == []


def test_clear_test_set_without_a_test_set_is_a_no_op():
    prompt_set = PromptSet(make_prompts(3))
    prompt_set.clear_test_set()
    assert len(prompt_set.get_train()) == 3


def test_getters_return_copies():
    prompt_set = PromptSet(make_prompts(3))
    prompt_set.get_train().clear()
    assert len(prompt_set.get_train()) == 3


# --------------------------------------------------------------------------
# map()
# --------------------------------------------------------------------------


def test_map_substitutes_doc_in_every_turn():
    prompt_set = PromptSet(
        [Prompt(chat_turns=["a {{DOC}}", "b {{DOC}}"], use_multiturn=True)]
    )
    mapped, _ = prompt_set.map(["SAMPLE"])
    assert mapped[0].chat_turns == ["a SAMPLE", "b SAMPLE"]


def test_map_returns_one_prompt_per_sample_cycling_templates():
    prompt_set = PromptSet(make_prompts(2))
    mapped, labels = prompt_set.map(["s0", "s1", "s2"])
    assert len(mapped) == 3
    assert len(labels) == 3
    assert mapped[0].chat_turns[0] == "p0t0:s0"
    assert mapped[1].chat_turns[0] == "p1t0:s1"
    assert mapped[2].chat_turns[0] == "p0t0:s2"


def test_map_does_not_mutate_the_template():
    templates = make_prompts(1)
    prompt_set = PromptSet(templates)
    mapped, _ = prompt_set.map(["SAMPLE"])
    mapped[0].metadata["extra"] = True
    mapped[0].examples.append(("u", "a"))
    assert templates[0].chat_turns[0] == "p0t0:{{DOC}}"
    assert "extra" not in templates[0].metadata
    assert templates[0].examples == []


def test_map_labels_keep_the_unsubstituted_template():
    # The label column records which template produced the row, so it must keep
    # the placeholder rather than the expanded document.
    prompt_set = PromptSet(make_prompts(1))
    _, labels = prompt_set.map(["SAMPLE"])
    assert labels[0]["chat_turns"] == ["p0t0:{{DOC}}"]
    assert labels[0]["use_multiturn"] is True
    assert labels[0]["metadata"] == {"PROMPT_TYPE": "type0"}


def test_map_labels_never_contain_an_empty_metadata_dict():
    # HF datasets cannot store an empty struct, so map() substitutes a stub.
    prompt_set = PromptSet([Prompt(chat_turns=["{{DOC}}"], use_multiturn=True)])
    _, labels = prompt_set.map(["s"])
    assert labels[0]["metadata"] == {"_dummy": True}


def test_map_from_the_test_split():
    prompt_set = PromptSet(make_prompts(4))
    prompt_set.generate_test_split(0.5)
    mapped, _ = prompt_set.map(["s"], use_test=True)
    assert mapped[0].chat_turns[0] == "p2t0:s"


def test_map_with_no_samples():
    prompt_set = PromptSet(make_prompts(2))
    mapped, labels = prompt_set.map([])
    assert mapped == []
    assert labels == []


# --------------------------------------------------------------------------
# shuffle
# --------------------------------------------------------------------------


def test_shuffle_is_seeded_and_resets_the_cursor():
    a = PromptSet(make_prompts(8))
    b = PromptSet(make_prompts(8))
    a.next_train(3)
    a.shuffle(seed=123)
    b.shuffle(seed=123)
    assert [p.chat_turns[0] for p in a.get_train()] == [
        p.chat_turns[0] for p in b.get_train()
    ]
    assert a.next_train(1)[0].chat_turns[0] == b.next_train(1)[0].chat_turns[0]


def test_shuffle_permutes_without_dropping_prompts():
    prompt_set = PromptSet(make_prompts(20))
    before = {p.chat_turns[0] for p in prompt_set.get_train()}
    prompt_set.shuffle(seed=7)
    after = [p.chat_turns[0] for p in prompt_set.get_train()]
    assert set(after) == before
    assert after != sorted(after, key=lambda s: int(s[1 : s.index("t")]))


def test_shuffle_leaves_the_test_set_alone():
    prompt_set = PromptSet(make_prompts(10))
    prompt_set.generate_test_split(0.3)
    test_before = [p.chat_turns[0] for p in prompt_set.get_test()]
    prompt_set.shuffle(seed=1)
    assert [p.chat_turns[0] for p in prompt_set.get_test()] == test_before


# --------------------------------------------------------------------------
# load_prompts
# --------------------------------------------------------------------------


def test_load_prompts_reads_all_fields(data_dir):
    prompts = load_prompts([str(data_dir / "prompts_valid.json")])
    assert len(prompts) == 2
    assert prompts[0].use_multiturn is True
    assert prompts[0].examples == [["Rewrite: hello", "greetings"]]
    assert prompts[0].metadata["PROMPT_TYPE"] == "rewrite"
    assert prompts[1].use_multiturn is False
    assert len(prompts[1].chat_turns) == 2


def test_load_prompts_concatenates_paths_in_order(data_dir):
    path = str(data_dir / "prompts_valid.json")
    prompts = load_prompts([path, path])
    assert len(prompts) == 4
    assert prompts[0].metadata == prompts[2].metadata


def test_load_prompts_rejects_non_json_paths(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Only JSON files"):
        load_prompts([str(path)])


def test_load_prompts_rejects_a_non_list_root(data_dir):
    with pytest.raises(ValueError, match="Expected a JSON list"):
        load_prompts([str(data_dir / "prompts_not_a_list.json")])


def test_load_prompts_rejects_non_object_entries(data_dir):
    with pytest.raises(ValueError, match="must be an object"):
        load_prompts([str(data_dir / "prompts_entry_not_object.json")])


def test_load_prompts_rejects_entries_missing_required_keys(data_dir):
    with pytest.raises(ValueError, match="must have"):
        load_prompts([str(data_dir / "prompts_missing_key.json")])


def test_load_prompts_roundtrips_through_the_prompt_set(data_dir):
    prompts = load_prompts([str(data_dir / "prompts_valid.json")])
    mapped, _ = PromptSet(prompts).map(["DOCUMENT"])
    assert mapped[0].chat_turns[0].endswith("DOCUMENT")


def test_load_prompts_accepts_a_file_written_by_json_dump(tmp_path):
    path = tmp_path / "generated.json"
    path.write_text(
        json.dumps(
            [
                {
                    "chat_turns": ["{{DOC}}"],
                    "use_multiturn": False,
                    "examples": [],
                    "metadata": {},
                }
            ]
        ),
        encoding="utf-8",
    )
    assert len(load_prompts([str(path)])) == 1
