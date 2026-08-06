from gen import clean_responses, rejection_reasons

BODY = " ".join(f"word{i}" for i in range(120))


# --------------------------------------------------------------------------
# clean_responses
# --------------------------------------------------------------------------


def test_the_response_is_put_into_the_source_whitespace_convention():
    assert clean_responses(["a  b\n\n\nc  \n"], ["a b\nc"]) == ["a b\nc"]


def test_a_wrapper_and_its_stray_whitespace_are_both_removed():
    assert clean_responses([f"Title\n\n---\n\n{BODY}"], ["plain source"]) == [BODY]


def test_a_leading_heading_is_kept_as_content():
    # Headings are usually real writing, not a task label.
    text = "**A Title Worth Keeping**\nThe body."
    assert clean_responses([text], ["The body."]) == [text]


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
# rejection_reasons
# --------------------------------------------------------------------------

REASONS = {"empty or too short", "refusal", "filler output",
           "unfilled placeholder", "task meta-commentary", "echoed instruction"}


def rejected(reasons):
    """Collapse the per-reason flags into one flag per row."""
    return [any(flags) for flags in zip(*reasons.values())]


def test_a_good_row_is_kept():
    reasons = rejection_reasons(["source"], [BODY], [""])
    assert rejected(reasons) == [False]
    assert sum(sum(f) for f in reasons.values()) == 0


def test_each_failure_mode_is_reported_under_its_own_reason():
    # Each response is long enough that only its own reason fires.
    originals = ["source"] * 4
    responses = [
        "",
        f"I cannot fulfill this request. {BODY}",
        f"The quick brown fox jumps over the lazy dog. {BODY}",
        f"{BODY}\n[Your Name]",
    ]
    reasons = rejection_reasons(originals, responses, [""] * len(originals))
    assert rejected(reasons) == [True, True, True, True]
    assert reasons["empty or too short"] == [True, False, False, False]
    assert reasons["refusal"] == [False, True, False, False]
    assert reasons["filler output"] == [False, False, True, False]
    assert reasons["unfilled placeholder"] == [False, False, False, True]


def test_reasons_may_overlap_on_one_row():
    reasons = rejection_reasons(["source"], ["I cannot fulfill this request."], [""])
    assert rejected(reasons) == [True]
    assert [n for n, f in reasons.items() if f[0]] == ["empty or too short", "refusal"]


def test_similarity_is_not_a_rejection_reason():
    # Over-similar pairs are the analysis stage's job, not this one.
    assert rejected(rejection_reasons([BODY], [BODY], [""])) == [False]


def test_an_empty_dataset_still_reports_every_reason():
    # The readme table needs stable keys even for an empty shard.
    reasons = rejection_reasons([], [], [])
    assert set(reasons) == REASONS
    assert rejected(reasons) == []


def test_the_source_column_is_never_handed_back_for_writing():
    # clean_responses returns only the AI column, so gen.py has nothing to
    # write back over `original`.
    result = clean_responses([f"Title\n---\n{BODY}"], ["**bold** source 🌱  \n\n"])
    assert isinstance(result, list) and len(result) == 1


# --------------------------------------------------------------------------
# The kept / trashed split, as main() performs it
# --------------------------------------------------------------------------


def split_like_main(rows):
    """Run main()'s post-processing surgery over an in-memory dataset."""
    from datasets import Dataset

    ds = Dataset.from_dict(rows)
    instructions = [""] * len(ds)
    originals = ds["original"]
    responses = clean_responses(ds["final_response"], originals)
    reasons = rejection_reasons(originals, responses, instructions)
    flags = [any(f) for f in zip(*reasons.values())]
    ds = ds.remove_columns("final_response").add_column("final_response", responses)
    labels = [", ".join(n for n, f in reasons.items() if f[i])
              for i, d in enumerate(flags) if d]
    trashed = ds.select([i for i, d in enumerate(flags) if d])
    if labels:
        trashed = trashed.add_column("rejected_for", labels)
    return ds.select([i for i, d in enumerate(flags) if not d]), trashed


ROWS = {
    "original": ["source one", "source two", "source three", "source four"],
    "final_response": [BODY, "I cannot fulfill this request. " + BODY,
                       BODY.upper(), f"{BODY}\n[Your Name]"],
    "marker": ["a", "b", "c", "d"],
}


def test_the_split_is_exhaustive_and_disjoint():
    kept, trashed = split_like_main(ROWS)
    assert len(kept) + len(trashed) == 4
    assert set(kept["marker"]).isdisjoint(trashed["marker"])


def test_rows_stay_aligned_with_their_own_source_text():
    kept, trashed = split_like_main(ROWS)
    for ds in (kept, trashed):
        for row in ds:
            expected = ROWS["original"][ROWS["marker"].index(row["marker"])]
            assert row["original"] == expected


def test_the_trashed_rows_carry_their_reasons():
    _, trashed = split_like_main(ROWS)
    labels = dict(zip(trashed["marker"], trashed["rejected_for"]))
    assert "refusal" in labels["b"]
    assert "unfilled placeholder" in labels["d"]


def test_only_the_trashed_side_gains_a_column():
    kept, trashed = split_like_main(ROWS)
    assert "rejected_for" not in kept.column_names
    assert "rejected_for" in trashed.column_names


def test_an_empty_trash_does_not_crash_add_column():
    # datasets raises "tables don't have the same number of rows" when a column
    # is added to an empty selection.
    kept, trashed = split_like_main({"original": ["s"], "final_response": [BODY], "marker": ["a"]})
    assert (len(kept), len(trashed)) == (1, 0)
