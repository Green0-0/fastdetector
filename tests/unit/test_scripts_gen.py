"""Post-processing of generated responses (``scripts/gen.py``).

Each rule strips a different flavour of LLM boilerplate, and rule 5 reverts the
whole thing when the strippers ate too much. The revert guard is why every
positive test below uses a long body: a short one would trip it.
"""

import pytest

from gen import post_process_response

BODY = " ".join(f"word{i}" for i in range(120))


def run(original: str, response: str) -> dict:
    """Post-process one row."""
    return post_process_response({"original": original, "final_response": response})


def mods(row: dict) -> tuple:
    """The five modification flags in order."""
    return (row["mod_1"], row["mod_2"], row["mod_3"], row["mod_4"], row["reverted"])


# --------------------------------------------------------------------------
# Rule 1: extract the content between a pair of horizontal rules
# --------------------------------------------------------------------------


def test_rule_1_extracts_the_text_between_two_rules():
    row = run("plain source document", f"Preamble\n---\n{BODY}\n---\nTrailer")
    assert row["final_response"].strip() == BODY
    assert mods(row) == (1, 0, 0, 0, 0)


def test_rule_1_does_not_apply_when_the_original_also_has_rules():
    # The source document legitimately contains "---", so the markers are not
    # boilerplate.
    response = f"Preamble\n---\n{BODY}\n---\nTrailer"
    row = run("source --- with a rule", response)
    assert row["final_response"] == response
    assert mods(row) == (0, 0, 0, 0, 0)


def test_rule_1_needs_exactly_two_rules():
    response = f"---\n{BODY}\n---\nmiddle\n---\n"
    row = run("plain", response)
    assert row["mod_1"] == 0


# --------------------------------------------------------------------------
# Rule 2: drop a trailing horizontal rule
# --------------------------------------------------------------------------


def test_rule_2_strips_a_trailing_rule():
    row = run("plain source", f"{BODY}\n---")
    assert row["final_response"].strip() == BODY
    assert mods(row) == (0, 1, 0, 0, 0)


def test_rule_2_leaves_a_trailing_rule_the_original_also_has():
    response = f"{BODY}\n---"
    row = run("source that ends with ---", response)
    assert row["final_response"] == response
    assert row["mod_2"] == 0


def test_rule_2_ignores_a_rule_in_the_middle():
    row = run("plain", f"{BODY}\n---\nmore text")
    assert row["mod_2"] == 0


# --------------------------------------------------------------------------
# Rule 3: drop an acknowledgement followed by a rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opener",
    [
        "Here is your rewrite:",
        "Sure, no problem:",
        "Certainly! Here we go:",
        "I'm happy to help with that:",
    ],
)
def test_rule_3_drops_an_acknowledgement_before_a_rule(opener):
    row = run("plain source", f"{opener}\n---\n{BODY}")
    assert row["final_response"].strip() == BODY
    assert row["mod_3"] == 1


def test_rule_3_is_case_insensitive():
    row = run("plain source", f"HERE IS THE TEXT:\n---\n{BODY}")
    assert row["mod_3"] == 1


def test_rule_3_leaves_an_acknowledgement_the_original_also_opens_with():
    response = f"Here is the text:\n---\n{BODY}"
    row = run("Here is the source document", response)
    assert row["mod_3"] == 0


def test_rule_3_needs_exactly_one_rule():
    row = run("plain source", f"Here is the text:\n{BODY}")
    assert row["mod_3"] == 0


# --------------------------------------------------------------------------
# Rule 4: drop a short acknowledgement line
# --------------------------------------------------------------------------


def test_rule_4_drops_a_short_leading_acknowledgement_line():
    row = run("plain source", f"Sure, here is the rewritten document:\n{BODY}")
    assert row["final_response"] == BODY
    assert mods(row) == (0, 0, 0, 1, 0)


def test_rule_4_keeps_a_long_first_line():
    # More than 20 words means the first line is probably real content.
    long_first_line = "Here " + " ".join(f"filler{i}" for i in range(30))
    response = f"{long_first_line}\n{BODY}"
    row = run("plain source", response)
    assert row["final_response"] == response
    assert row["mod_4"] == 0


def test_rule_4_needs_a_newline_to_cut_at():
    response = f"Here is the rewrite: {BODY}"
    row = run("plain source", response)
    assert row["final_response"] == response
    assert row["mod_4"] == 0


def test_rule_4_also_strips_a_hanging_rule_left_behind():
    # Two rules plus a "---" in the original: rules 1-3 all decline, so rule 4
    # is what removes the acknowledgement and the orphaned separator.
    row = run("source --- document", f"Here you go:\n---\n{BODY}\n---\ntail")
    assert row["final_response"].startswith("word0")
    assert row["mod_4"] == 1


def test_rule_4_leaves_an_acknowledgement_the_original_also_opens_with():
    response = f"Here is the rewrite:\n{BODY}"
    row = run("Here, the source starts the same way", response)
    assert row["mod_4"] == 0


# --------------------------------------------------------------------------
# Rule 5: revert when too much was removed
# --------------------------------------------------------------------------


def test_a_response_losing_more_than_a_quarter_of_its_words_is_reverted():
    response = "Here you go:\nshort body"
    row = run("plain source", response)
    assert row["final_response"] == response
    assert mods(row) == (0, 0, 0, 0, 1)


def test_a_response_losing_more_than_forty_words_is_reverted():
    # 52 words dropped out of 402 is only 13%, so the absolute cap is what
    # triggers here.
    preamble = " ".join(f"pre{i}" for i in range(45))
    tail = " ".join(f"tail{i}" for i in range(5))
    long_body = " ".join(f"body{i}" for i in range(350))
    row = run("plain source", f"{preamble}\n---\n{long_body}\n---\n{tail}")
    assert row["reverted"] == 1
    assert row["final_response"].startswith("pre0")


def test_reverting_clears_every_modification_flag():
    row = run("plain source", "Sure:\nshort")
    assert mods(row) == (0, 0, 0, 0, 1)


def test_a_small_removal_is_kept():
    row = run("plain source", f"Here you go:\n{BODY}")
    assert row["reverted"] == 0
    assert row["final_response"] == BODY


# --------------------------------------------------------------------------
# General contract
# --------------------------------------------------------------------------


def test_a_clean_response_is_untouched():
    row = run("plain source", BODY)
    assert row["final_response"] == BODY
    assert mods(row) == (0, 0, 0, 0, 0)


def test_every_tracking_column_is_always_present():
    row = run("plain source", BODY)
    for column in ("mod_1", "mod_2", "mod_3", "mod_4", "reverted"):
        assert column in row


def test_the_original_column_is_never_modified():
    row = run("plain source", f"Here you go:\n{BODY}")
    assert row["original"] == "plain source"


def test_an_empty_response_is_handled():
    row = run("plain source", "")
    assert row["final_response"] == ""
    assert mods(row) == (0, 0, 0, 0, 0)


def test_an_empty_original_is_handled():
    row = run("", f"Here you go:\n{BODY}")
    assert row["final_response"] == BODY


def test_missing_keys_default_to_empty_strings():
    row = post_process_response({})
    assert row["final_response"] == ""
    assert row["reverted"] == 0


def test_extra_columns_are_preserved():
    row = post_process_response(
        {"original": "plain", "final_response": BODY, "prompt": {"metadata": {}}}
    )
    assert row["prompt"] == {"metadata": {}}
