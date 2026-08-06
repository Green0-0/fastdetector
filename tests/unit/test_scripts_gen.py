from gen import clean_columns, prompt_instructions, rejected_rows

BODY = " ".join(f"word{i}" for i in range(120))


# --------------------------------------------------------------------------
# clean_columns
# --------------------------------------------------------------------------


def test_both_columns_end_up_in_the_same_whitespace_convention():
    originals, responses = clean_columns(["a  b\nc"], ["a  b\n\n\nc  \n"])
    assert originals == responses == ["a b\nc"]


def test_a_wrapper_and_its_stray_whitespace_are_both_removed():
    _, responses = clean_columns(["plain source"], [f"Title\n\n---\n\n{BODY}"])
    assert responses == [BODY]


def test_an_added_title_is_removed():
    _, responses = clean_columns(["The body."], ["**Recreated Text:**\n\nThe body."])
    assert responses == ["The body."]


def test_markdown_and_emoji_are_stripped_from_both_columns():
    originals, responses = clean_columns(["**bold** human 🌱"], ["Lead line.\n## Heading\n**bold** ai 🤝"])
    assert originals == ["bold human"]
    assert responses == ["Lead line.\nHeading\nbold ai"]


def test_a_title_is_removed_before_markdown_stripping_erases_the_evidence():
    # strip_markdown would delete the ** that marks the title as a title.
    _, responses = clean_columns(["Real content here."], ["**Rough Draft:**\n\nReal content here."])
    assert responses == ["Real content here."]


def test_mojibake_is_repaired_in_both_columns():
    originals, responses = clean_columns(["cÃ³mo"], ["estÃ¡s"])
    assert (originals, responses) == (["cómo"], ["estás"])


def test_a_clean_pair_survives_untouched():
    originals, responses = clean_columns(["source text"], [BODY])
    assert (originals, responses) == (["source text"], [BODY])


def test_the_columns_stay_aligned():
    originals, responses = clean_columns(["a", "b", "c"], ["x", "y", "z"])
    assert len(originals) == len(responses) == 3


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
