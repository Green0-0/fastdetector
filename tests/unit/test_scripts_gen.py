from gen import clean_responses, rejected_rows

BODY = " ".join(f"word{i}" for i in range(120))


# --------------------------------------------------------------------------
# clean_responses
# --------------------------------------------------------------------------


def test_the_response_is_put_into_the_source_whitespace_convention():
    assert clean_responses(["a  b\n\n\nc  \n"], ["a b\nc"]) == ["a b\nc"]


def test_a_wrapper_and_its_stray_whitespace_are_both_removed():
    assert clean_responses([f"Title\n\n---\n\n{BODY}"], ["plain source"]) == [BODY]


def test_an_added_title_is_removed():
    assert clean_responses(["**Recreated Text:**\n\nThe body."], ["The body."]) == ["The body."]


def test_mojibake_is_repaired_in_the_response():
    assert clean_responses(["estÃ¡s"], ["source"]) == ["estás"]


def test_markdown_and_emoji_are_left_alone():
    # Stripping them from the AI side only would make them a human signal.
    text = "**bold** and an emoji 🤝"
    assert clean_responses([text], ["plain source"]) == [text]


def test_a_clean_response_survives_untouched():
    assert clean_responses([BODY], ["source text"]) == [BODY]


def test_the_output_stays_aligned_with_the_input():
    assert len(clean_responses(["x", "y", "z"], ["a", "b", "c"])) == 3


# --------------------------------------------------------------------------
# rejected_rows
# --------------------------------------------------------------------------


def test_a_good_row_is_kept():
    rejected, counts = rejected_rows(["source"], [BODY], [""])
    assert rejected == [False]
    assert sum(counts.values()) == 0


def test_each_failure_mode_is_counted_under_its_own_reason():
    # Each response is long enough that only its own reason fires.
    originals = ["source"] * 4
    responses = [
        "",
        f"I cannot fulfill this request. {BODY}",
        f"The quick brown fox jumps over the lazy dog. {BODY}",
        f"{BODY}\n[Your Name]",
    ]
    rejected, counts = rejected_rows(originals, responses, [""] * len(originals))
    assert rejected == [True, True, True, True]
    assert counts["empty or too short"] == 1
    assert counts["refusal"] == 1
    assert counts["filler output"] == 1
    assert counts["unfilled placeholder"] == 1


def test_reasons_may_overlap_on_one_row():
    rejected, counts = rejected_rows(["source"], ["I cannot fulfill this request."], [""])
    assert rejected == [True]
    assert sum(counts.values()) == 2  # too short *and* a refusal


def test_similarity_is_not_a_rejection_reason():
    # Over-similar pairs are the analysis stage's job, not this one.
    rejected, _ = rejected_rows([BODY], [BODY], [""])
    assert rejected == [False]


def test_an_empty_dataset_still_reports_every_reason():
    # The readme table needs stable keys even for an empty shard.
    rejected, counts = rejected_rows([], [], [])
    assert rejected == []
    assert set(counts) == {"empty or too short", "refusal", "filler output",
                           "unfilled placeholder", "task meta-commentary", "echoed instruction"}
    assert sum(counts.values()) == 0


def test_the_source_column_is_never_handed_back_for_writing():
    # clean_responses returns only the AI column, so gen.py has nothing to
    # write back over `original`.
    result = clean_responses([f"Title\n---\n{BODY}"], ["**bold** source 🌱  \n\n"])
    assert isinstance(result, list) and len(result) == 1
