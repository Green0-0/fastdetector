import json

import pytest

from fastdetector.prompting.prompt_builder import (
    add_example,
    add_metadata,
    apply_recursive_format,
    force_reformat,
    generate_dataset,
    load_raw_samples,
    load_raw_samples_balanced_autosplit,
    partial_stack,
    resize,
    save_dataset,
    shuffle,
    split,
)
from fastdetector.prompting.prompts import Prompt, load_prompts


def items(n: int) -> list[list[str]]:
    """Build ``n`` single-turn chat lists with identifiable contents."""
    return [[f"item{i}"] for i in range(n)]


# --------------------------------------------------------------------------
# shuffle
# --------------------------------------------------------------------------


def test_shuffle_does_not_mutate_the_input():
    """Test that shuffle does not mutate original input list in place."""
    original = items(10)
    snapshot = [list(chat) for chat in original]
    shuffle(original, seed=1)
    assert original == snapshot


def test_shuffle_is_deterministic_for_a_seed():
    """Test that shuffle produces identical ordering given identical seed."""
    assert shuffle(items(10), seed=5) == shuffle(items(10), seed=5)


def test_shuffle_changes_order_and_preserves_membership():
    """Test that shuffle changes element ordering while preserving set membership."""
    result = shuffle(items(30), seed=3)
    assert sorted(result) == sorted(items(30))
    assert result != items(30)


def test_shuffle_of_empty_list():
    """Test that shuffle handles empty list input."""
    assert shuffle([], seed=1) == []


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def test_split_at_half():
    """Test splitting a list equally in half."""
    first, second = split(items(10), proportion=0.5, min_size=1)
    assert len(first) == 5
    assert len(second) == 5
    assert first + second == items(10)


def test_split_clamps_to_min_size_on_the_high_side():
    """Test split clamps output list size to respect min_size upper bound."""
    first, second = split(items(10), proportion=0.99, min_size=3)
    assert len(first) == 7
    assert len(second) == 3


def test_split_clamps_to_min_size_on_the_low_side():
    """Test split clamps output list size to respect min_size lower bound."""
    first, second = split(items(10), proportion=0.0, min_size=2)
    assert len(first) == 2
    assert len(second) == 8


def test_split_rounds_rather_than_truncates():
    """Test split rounds fractional sizes to nearest integer."""
    first, _ = split(items(10), proportion=0.26, min_size=1)
    assert len(first) == 3


def test_split_rejects_an_empty_list():
    """Test that split raises AssertionError when input list is empty."""
    with pytest.raises(AssertionError, match="empty list"):
        split([], proportion=0.5, min_size=1)


def test_split_rejects_an_unsatisfiable_min_size():
    """Test that split raises AssertionError when min_size cannot be satisfied."""
    with pytest.raises(AssertionError, match="min_size"):
        split(items(3), proportion=0.5, min_size=2)


def test_split_returns_copies():
    """Test that split returns shallow copies of sublists."""
    source = items(4)
    first, _ = split(source, proportion=0.5, min_size=1)
    first.clear()
    assert len(source) == 4


# --------------------------------------------------------------------------
# resize
# --------------------------------------------------------------------------


def test_resize_down_returns_a_subset():
    """Test resizing list down to smaller target length."""
    result = resize(items(10), target_length=4, also_shuffle=False)
    assert result == items(4)


def test_resize_down_with_shuffle_keeps_membership():
    """Test resizing down with shuffle maintains element membership."""
    result = resize(items(10), target_length=4, also_shuffle=True, seed=2)
    assert len(result) == 4
    assert all(entry in items(10) for entry in result)


def test_resize_up_duplicates_as_evenly_as_possible():
    """Test resizing list up duplicates items evenly across target length."""
    result = resize(items(3), target_length=7, also_shuffle=False)
    assert len(result) == 7
    counts = {entry[0]: result.count(entry) for entry in items(3)}
    # 7 = 2 full copies of 3 + 1 sampled extra, so nothing is used 3+ times
    # more than anything else.
    assert min(counts.values()) == 2
    assert max(counts.values()) == 3


def test_resize_up_to_an_exact_multiple_uses_every_item_equally():
    """Test resizing up to exact multiple replicates every item equally."""
    result = resize(items(4), target_length=12, also_shuffle=False)
    assert len(result) == 12
    for entry in items(4):
        assert result.count(entry) == 3


def test_resize_to_the_same_length_is_a_copy():
    """Test resizing to identical target length returns list copy."""
    assert resize(items(5), target_length=5, also_shuffle=False) == items(5)


def test_resize_does_not_mutate_the_input():
    """Test that resize leaves input list unmutated."""
    source = items(3)
    resize(source, target_length=9)
    assert source == items(3)


def test_resize_rejects_an_empty_source():
    """Test that resize raises AssertionError for empty source list."""
    with pytest.raises(AssertionError, match="empty list"):
        resize([], target_length=5)


def test_resize_rejects_a_non_positive_target():
    """Test that resize raises AssertionError for non-positive target length."""
    with pytest.raises(AssertionError, match="target_length"):
        resize(items(3), target_length=0)


# --------------------------------------------------------------------------
# partial_stack
# --------------------------------------------------------------------------


def test_partial_stack_with_a_fixed_size_of_one_takes_only_the_first_set():
    """Test partial_stack with stack_size=1 takes elements only from first set."""
    a = [["a0"], ["a1"]]
    b = [["b0"], ["b1"]]
    assert partial_stack([a, b], min_stack_size=1, max_stack_size=1) == a


def test_partial_stack_with_a_fixed_size_of_two_concatenates_row_wise():
    """Test partial_stack with stack_size=2 concatenates items row-wise."""
    a = [["a0"], ["a1"]]
    b = [["b0"], ["b1"]]
    assert partial_stack([a, b], min_stack_size=2, max_stack_size=2) == [
        ["a0", "b0"],
        ["a1", "b1"],
    ]


def test_partial_stack_varies_length_between_the_bounds():
    """Test partial_stack samples stack lengths randomly within specified bounds."""
    a, b, c = ([[f"{n}{i}"] for i in range(50)] for n in "abc")
    result = partial_stack([a, b, c], min_stack_size=1, max_stack_size=3, seed=1)
    lengths = {len(row) for row in result}
    assert len(result) == 50
    assert lengths <= {1, 2, 3}
    assert len(lengths) > 1
    for i, row in enumerate(result):
        assert row[0] == f"a{i}"


def test_partial_stack_rejects_ragged_inputs():
    """Test that partial_stack raises AssertionError for inputs with differing row counts."""
    with pytest.raises(AssertionError, match="same length"):
        partial_stack([[["a0"]], [["b0"], ["b1"]]])


def test_partial_stack_rejects_an_empty_input():
    """Test that partial_stack raises AssertionError when given empty input."""
    with pytest.raises(AssertionError, match="cannot be empty"):
        partial_stack([])


def test_partial_stack_rejects_a_max_larger_than_the_number_of_sets():
    """Test that partial_stack raises AssertionError when max_stack_size exceeds set count."""
    with pytest.raises(AssertionError, match="max_stack_size"):
        partial_stack([[["a"]]], min_stack_size=1, max_stack_size=2)


def test_partial_stack_rejects_inverted_bounds():
    """Test that partial_stack raises AssertionError for min_stack_size > max_stack_size."""
    with pytest.raises(AssertionError, match="Invalid bounds"):
        partial_stack([[["a"]], [["b"]]], min_stack_size=2, max_stack_size=1)


# --------------------------------------------------------------------------
# force_reformat
# --------------------------------------------------------------------------


def test_force_reformat_wraps_every_message():
    """Test force_reformat applies template format to every chat message."""
    result = force_reformat([["one", "two"]], modified_format="<{{TEXT}}>")
    assert result == [["<one>", "<two>"]]


def test_force_reformat_can_target_only_the_first_message():
    """Test force_reformat with only_first_message=True formats only the first message."""
    result = force_reformat(
        [["one", "two"]], only_first_message=True, modified_format="<{{TEXT}}>"
    )
    assert result == [["<one>", "two"]]


def test_force_reformat_requires_the_placeholder():
    """Test force_reformat raises AssertionError if format template omits {{TEXT}}."""
    with pytest.raises(AssertionError, match=r"\{\{TEXT\}\}"):
        force_reformat([["one"]], modified_format="no placeholder")


def test_force_reformat_does_not_mutate_the_input():
    """Test that force_reformat returns new formatted list without mutating original."""
    source = [["one"]]
    force_reformat(source, modified_format="<{{TEXT}}>")
    assert source == [["one"]]


# --------------------------------------------------------------------------
# apply_recursive_format
# --------------------------------------------------------------------------


def test_apply_recursive_format_leaves_the_first_turn_alone():
    """Test apply_recursive_format leaves first chat turn unformatted."""
    result = apply_recursive_format([["first", "second", "third"]])
    assert result[0][0] == "first"


def test_apply_recursive_format_numbers_headers_from_zero():
    """Test apply_recursive_format inserts incrementing response placeholders."""
    result = apply_recursive_format([["first", "second", "third"]], order="first")
    assert result[0][1] == "{{RESP_0}}\nsecond"
    assert result[0][2] == "{{RESP_1}}\nthird"


def test_apply_recursive_format_can_append_the_header_last():
    """Test apply_recursive_format appends response header at end when order='last'."""
    result = apply_recursive_format([["first", "second"]], order="last")
    assert result[0][1] == "second\n{{RESP_0}}"


def test_apply_recursive_format_honours_a_custom_header():
    """Test apply_recursive_format accepts custom response header templates."""
    result = apply_recursive_format([["a", "b"]], res_header="<PREV_#>")
    assert result[0][1] == "<PREV_0>\nb"


def test_apply_recursive_format_rejects_an_unknown_order():
    """Test apply_recursive_format raises AssertionError for invalid order value."""
    with pytest.raises(AssertionError, match="order must be"):
        apply_recursive_format([["a", "b"]], order="middle")


def test_apply_recursive_format_rejects_empty_chats():
    """Test apply_recursive_format raises AssertionError for empty chat entries."""
    with pytest.raises(AssertionError, match="cannot be empty"):
        apply_recursive_format([[]])


def test_apply_recursive_format_placeholders_match_what_the_generator_replaces():
    """Test that response placeholders match generator substitution format."""
    # The generator substitutes literally "{{RESP_<n>}}"; a mismatch here would
    # silently ship un-substituted placeholders to the model.
    from fastdetector.generator import _build_messages

    chat = apply_recursive_format([["first", "second"]])[0]
    prompt = Prompt(chat_turns=chat, use_multiturn=True)
    messages = _build_messages(prompt, turn_index=1, responses=["ANSWER0"])
    assert messages[-1]["content"] == "ANSWER0\nsecond"


# --------------------------------------------------------------------------
# Raw sample IO
# --------------------------------------------------------------------------


def test_load_raw_samples_wraps_each_string_in_its_own_chat(data_dir):
    """Test load_raw_samples wraps each sample string in single-turn chat list."""
    samples = load_raw_samples([str(data_dir / "raw_samples_a.json")])
    assert samples == [["sample a1"], ["sample a2"], ["sample a3"], ["sample a4"]]


def test_load_raw_samples_concatenates_paths(data_dir):
    """Test load_raw_samples concatenates samples across multiple file paths."""
    samples = load_raw_samples(
        [str(data_dir / "raw_samples_a.json"), str(data_dir / "raw_samples_b.json")]
    )
    assert len(samples) == 8
    assert samples[4] == ["sample b1"]


def test_load_raw_samples_rejects_a_missing_file(tmp_path):
    """Test load_raw_samples raises AssertionError when target file does not exist."""
    with pytest.raises(AssertionError, match="File not found"):
        load_raw_samples([str(tmp_path / "nope.json")])


def test_load_raw_samples_rejects_a_non_list_root(data_dir):
    """Test load_raw_samples raises AssertionError if JSON root is not a list."""
    with pytest.raises(AssertionError, match="must be a list"):
        load_raw_samples([str(data_dir / "raw_samples_not_a_list.json")])


def test_load_raw_samples_rejects_non_string_items(data_dir):
    """Test load_raw_samples raises AssertionError if JSON list contains non-string items."""
    with pytest.raises(AssertionError, match="unsupported type"):
        load_raw_samples([str(data_dir / "raw_samples_bad_type.json")])


def test_balanced_autosplit_splits_each_path_independently(data_dir):
    """Test load_raw_samples_balanced_autosplit splits each input file independently."""
    first, second = load_raw_samples_balanced_autosplit(
        [str(data_dir / "raw_samples_a.json"), str(data_dir / "raw_samples_b.json")],
        split_proportion=0.5,
        min_size=1,
    )
    assert first == [["sample a1"], ["sample a2"], ["sample b1"], ["sample b2"]]
    assert second == [["sample a3"], ["sample a4"], ["sample b3"], ["sample b4"]]


def test_balanced_autosplit_can_shuffle_before_splitting(data_dir):
    """Test load_raw_samples_balanced_autosplit shuffles before splitting when requested."""
    paths = [str(data_dir / "raw_samples_a.json"), str(data_dir / "raw_samples_b.json")]
    first, second = load_raw_samples_balanced_autosplit(
        paths, split_proportion=0.5, min_size=1, shuffle_before_split=True, seed=1
    )
    assert len(first) == 4
    assert len(second) == 4
    assert not set(map(tuple, first)) & set(map(tuple, second))


def test_balanced_autosplit_uses_a_different_seed_per_path(data_dir):
    """Test load_raw_samples_balanced_autosplit varies PRNG seed per path."""
    # seed + i keeps two identical files from being split identically.
    path = str(data_dir / "raw_samples_a.json")
    first, _ = load_raw_samples_balanced_autosplit(
        [path, path], split_proportion=0.5, min_size=1, shuffle_before_split=True, seed=1
    )
    assert first[:2] != first[2:]


# --------------------------------------------------------------------------
# Prompt object construction
# --------------------------------------------------------------------------


def test_generate_dataset_builds_prompts():
    """Test generate_dataset constructs Prompt objects from chat lists."""
    prompts = generate_dataset([["a"], ["b"]], use_multiturn=False)
    assert len(prompts) == 2
    assert all(isinstance(p, Prompt) for p in prompts)
    assert prompts[0].chat_turns == ["a"]
    assert prompts[0].use_multiturn is False


def test_add_metadata_applies_to_every_prompt():
    """Test add_metadata attaches key-value pairs to every Prompt in list."""
    prompts = add_metadata(generate_dataset([["a"], ["b"]]), "PROMPT_TYPE", "rewrite")
    assert [p.metadata["PROMPT_TYPE"] for p in prompts] == ["rewrite", "rewrite"]


def test_add_metadata_keeps_prompt_dicts_independent():
    """Test add_metadata maintains independent metadata dictionaries across prompts."""
    prompts = generate_dataset([["a"], ["b"]])
    add_metadata(prompts, "k", 1)
    prompts[0].metadata["only_first"] = True
    assert "only_first" not in prompts[1].metadata


def test_add_example_appends_to_every_prompt():
    """Test add_example appends example tuple to every Prompt object."""
    prompts = add_example(generate_dataset([["a"], ["b"]]), ("user", "assistant"))
    assert prompts[0].examples == [("user", "assistant")]
    assert prompts[1].examples == [("user", "assistant")]


def test_add_example_twice_accumulates():
    """Test add_example accumulates multiple example tuples."""
    prompts = generate_dataset([["a"]])
    add_example(prompts, ("u1", "a1"))
    add_example(prompts, ("u2", "a2"))
    assert len(prompts[0].examples) == 2


# --------------------------------------------------------------------------
# save_dataset
# --------------------------------------------------------------------------


def test_save_dataset_writes_a_file_load_prompts_can_read(tmp_path):
    """Test save_dataset serializes prompts to JSON file readable by load_prompts."""
    prompts = add_metadata(
        add_example(generate_dataset([["a {{DOC}}"], ["b {{DOC}}"]]), ("u", "a")),
        "PROMPT_TYPE",
        "rewrite",
    )
    save_dataset(prompts, "my_set", path=str(tmp_path))

    out = tmp_path / "my_set.json"
    assert out.is_file()
    reloaded = load_prompts([str(out)])
    assert len(reloaded) == 2
    assert reloaded[0].chat_turns == ["a {{DOC}}"]
    assert reloaded[0].metadata == {"PROMPT_TYPE": "rewrite"}
    assert reloaded[0].examples == [["u", "a"]]


def test_save_dataset_does_not_double_the_extension(tmp_path):
    """Test save_dataset avoids duplicate .json extensions if already provided."""
    save_dataset(generate_dataset([["a"]]), "named.json", path=str(tmp_path))
    assert (tmp_path / "named.json").is_file()
    assert not (tmp_path / "named.json.json").exists()


def test_save_dataset_creates_missing_directories(tmp_path):
    """Test save_dataset creates parent output directories automatically."""
    target = tmp_path / "nested" / "deeper"
    save_dataset(generate_dataset([["a"]]), "set", path=str(target))
    assert (target / "set.json").is_file()


def test_save_dataset_rejects_an_empty_dataset(tmp_path):
    """Test that save_dataset raises AssertionError for empty prompt list."""
    with pytest.raises(AssertionError, match="empty dataset"):
        save_dataset([], "set", path=str(tmp_path))


def test_save_dataset_output_is_pretty_printed_json(tmp_path):
    """Test save_dataset formats JSON with pretty-printed indentation."""
    save_dataset(generate_dataset([["a"]]), "set", path=str(tmp_path))
    raw = (tmp_path / "set.json").read_text(encoding="utf-8")
    assert "\n    " in raw
    assert json.loads(raw)[0]["use_multiturn"] is True
