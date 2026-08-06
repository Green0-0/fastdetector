import pytest

from fastdetector.statistics.filters import (
    fix_encoding,
    has_encoding_damage,
    has_filler_output,
    has_instruction_vocabulary,
    has_length_anomaly,
    has_meta_commentary,
    has_placeholder,
    has_prompt_echo,
    has_refusal,
    is_duplicate,
    is_empty,
    is_near_duplicate,
    normalize_whitespace,
    strip_added_title,
    strip_emoji,
    strip_markdown,
    strip_wrapper_boilerplate,
)

BODY = " ".join(f"word{i}" for i in range(120))


def clean(original: str, response: str) -> str:
    """Post-process one response against its source text."""
    return strip_wrapper_boilerplate([response], [original])[0]


# --------------------------------------------------------------------------
# Issue 1: whitespace normalization
# --------------------------------------------------------------------------


def test_horizontal_whitespace_runs_collapse_to_one_space():
    assert normalize_whitespace(["a  b\tc"]) == ["a b c"]


def test_blank_lines_and_edge_whitespace_are_removed():
    assert normalize_whitespace(["\n\n a \n\n\n b \n"]) == ["a\nb"]


def test_a_space_before_a_newline_never_survives():
    assert normalize_whitespace(["line one  \nline two"]) == ["line one\nline two"]


def test_already_normalized_text_is_untouched():
    text = "line one\nline two"
    assert normalize_whitespace([text]) == [text]


def test_missing_values_become_empty_strings():
    assert normalize_whitespace([None]) == [""]


# --------------------------------------------------------------------------
# Issue 2: wrapper boilerplate
# --------------------------------------------------------------------------


def test_the_text_between_two_rules_is_extracted():
    assert clean("plain source document", f"Preamble\n---\n{BODY}\n---\nTrailer") == BODY


def test_rules_the_original_also_uses_are_left_alone():
    response = f"Preamble\n---\n{BODY}\n---\nTrailer"
    assert clean("source --- with a rule", response) == response


def test_a_trailing_rule_is_dropped():
    assert clean("plain source", f"{BODY}\n---") == BODY


def test_a_long_trailer_is_kept_instead_of_being_dropped():
    # The old rule 1 discarded everything after the second rule; a real closing
    # paragraph has to survive.
    trailer = " ".join(f"tail{i}" for i in range(30))
    assert clean("plain source", f"Title\n---\n{BODY}\n---\n{trailer}").endswith(trailer)


@pytest.mark.parametrize(
    "opener",
    [
        "Here is your rewrite:",
        "Sure, no problem:",
        "Certainly! Here we go:",
        "I'm happy to help with that:",
    ],
)
def test_an_acknowledgement_before_a_rule_is_dropped(opener):
    assert clean("plain source", f"{opener}\n---\n{BODY}") == BODY


def test_acknowledgements_are_matched_case_insensitively():
    assert clean("plain source", f"HERE IS THE TEXT:\n---\n{BODY}") == BODY


def test_an_acknowledgement_the_original_also_opens_with_is_kept():
    response = f"Here is the text:\n{BODY}"
    assert clean("Here is the source document", response) == response


def test_a_rule_still_marks_a_wrapper_even_after_a_shared_opening():
    # The acknowledgement guard only covers the bare opening line; a horizontal
    # rule is wrapper evidence in its own right.
    assert clean("Here is the source document", f"Here is the text:\n---\n{BODY}") == BODY


def test_a_short_acknowledgement_line_is_dropped_without_a_rule():
    assert clean("plain source", f"Sure, here is the rewritten document:\n{BODY}") == BODY


def test_a_long_first_line_is_treated_as_content():
    long_first_line = "Here " + " ".join(f"filler{i}" for i in range(30))
    response = f"{long_first_line}\n{BODY}"
    assert clean("plain source", response) == response


def test_an_acknowledgement_with_no_newline_to_cut_at_is_kept():
    response = f"Here is the rewrite: {BODY}"
    assert clean("plain source", response) == response


def test_a_response_losing_more_than_a_quarter_of_its_words_is_reverted():
    response = "Here you go:\nshort body"
    assert clean("plain source", response) == response


def test_a_response_losing_more_than_forty_words_is_reverted():
    preamble = " ".join(f"pre{i}" for i in range(45))
    long_body = " ".join(f"body{i}" for i in range(350))
    assert clean("plain source", f"{preamble}\n---\n{long_body}").startswith("pre0")


def test_the_result_never_keeps_edge_whitespace():
    cleaned = clean("plain source", f"Title\n\n---\n\n{BODY}\n\n")
    assert cleaned == cleaned.strip()


def test_a_clean_response_is_untouched():
    assert clean("plain source", BODY) == BODY


def test_empty_inputs_are_handled():
    assert clean("plain source", "") == ""
    assert clean("", f"Here you go:\n{BODY}") == BODY


# --------------------------------------------------------------------------
# Issue 8: formatting leakage
# --------------------------------------------------------------------------


def test_an_added_bold_title_line_is_removed():
    assert strip_added_title(["**Recreated Text:**\n\nThe body.", ], ["The body."]) == ["The body."]


def test_a_rule_hanging_under_a_removed_title_goes_with_it():
    assert strip_added_title(["**Rough Draft:**\n---\nThe body."], ["The body."]) == ["The body."]


def test_a_title_the_original_also_has_is_kept():
    text = "# Heading\n\nThe body."
    assert strip_added_title([text], ["# Heading\n\nSomething else."]) == [text]


def test_markdown_emphasis_headings_and_fences_are_removed():
    assert strip_markdown(["## Title\n**bold** and `code`"]) == ["Title\nbold and code"]


def test_markdown_links_keep_their_anchor_text():
    assert strip_markdown(["see [the docs](http://example.com)"]) == ["see the docs"]


def test_list_markers_are_preserved():
    text = "- first\n- second"
    assert strip_markdown([text]) == [text]


def test_emoji_are_removed():
    assert strip_emoji(["Hey there! 🌱 Welcome"]) == ["Hey there!  Welcome"]


# --------------------------------------------------------------------------
# Issue 14: encoding
# --------------------------------------------------------------------------


def test_mojibake_is_re_decoded():
    assert fix_encoding(["cÃ³mo estÃ¡s"]) == ["cómo estás"]


def test_literal_escape_sequences_become_real_whitespace():
    assert fix_encoding(["one\\ntwo"]) == ["one\ntwo"]


def test_clean_text_survives_re_decoding_untouched():
    assert fix_encoding(["plain ascii"]) == ["plain ascii"]


def test_replacement_characters_are_flagged():
    assert has_encoding_damage(["a�b", "clean"]) == [True, False]


# --------------------------------------------------------------------------
# Boolean row filters
# --------------------------------------------------------------------------


def test_empty_and_short_responses_are_flagged():
    assert is_empty(["", "   ", "too short", BODY]) == [True, True, True, False]


def test_the_length_floor_is_configurable():
    assert is_empty(["abc"], min_characters=2) == [False]


@pytest.mark.parametrize(
    "text",
    [
        "I cannot fulfill this request.",
        "I'm unable to generate the requested text because you haven't provided it.",
        "As an AI, I must decline.",
        "Please provide the source text.",
        "It appears that the input provided is a placeholder with no actual content.",
        "There doesn't appear to be any actual text or descriptor in your message.",
    ],
)
def test_refusals_are_flagged(text):
    assert has_refusal([text]) == [True]


@pytest.mark.parametrize(
    "text",
    [
        "I will not let negative people think that I can't do it.",
        "Unfortunately, the official update is not accessible to all users yet.",
        "There is no legitimate rationale for denying people the right to marry.",
    ],
)
def test_a_refusal_phrase_inside_ordinary_prose_is_not_a_refusal(text):
    # A complaint opener only counts when it is complaining about the input.
    assert has_refusal([text]) == [False]


def test_pangram_and_tag_filler_is_flagged():
    texts = ["The quick brown fox jumps over the lazy dog.", "[NOUN] [VERB]", BODY]
    assert has_filler_output(texts, ["source"] * 3) == [True, True, False]


def test_filler_the_original_discusses_is_not_flagged():
    assert has_filler_output(["An article about Lorem Ipsum."], ["Lorem Ipsum explained."]) == [False]


def test_unfilled_placeholders_are_flagged():
    assert has_placeholder(["Regards,\n[Your Name]", BODY], ["source", "source"]) == [True, False]


def test_a_placeholder_the_original_also_has_is_not_flagged():
    assert has_placeholder(["[Your Name]"], ["signed [Your Name]"]) == [False]


@pytest.mark.parametrize(
    "text",
    [
        "Reach us at [email protected] any time.",   # scraped address obfuscation
        "Turnout rose sharply that year. [Source: Reuters]",
    ],
)
def test_bracketed_web_text_is_not_a_placeholder(text):
    assert has_placeholder([text], ["source"]) == [False]


def test_task_meta_commentary_is_flagged():
    assert has_meta_commentary(["Note: the original text was formal."], ["source"]) == [True]


def test_meta_wording_the_original_uses_is_not_flagged():
    text = "the original text of the treaty"
    assert has_meta_commentary([text], [text]) == [False]


@pytest.mark.parametrize(
    "text",
    [
        "Please paste the following text into the terminal.",  # commoner in human prose
        "Candidates must bring the following document to the interview.",
        "As requested, the shipment left on Tuesday.",         # barely discriminates
    ],
)
def test_ordinary_prose_is_not_meta_commentary(text):
    assert has_meta_commentary([text], ["source"]) == [False]


def test_a_response_extrapolating_the_instructions_example_is_flagged():
    # Nothing is copied verbatim, so has_prompt_echo cannot see this.
    instruction = "Replace all nouns and verbs with their tags (e.g., [NOUN], [VERB])."
    response = "[NOUN] [VERB] [ADJECTIVE] [CONJUNCTION]"
    assert has_prompt_echo([response], [instruction]) == [False]
    assert has_instruction_vocabulary([response], [instruction]) == [True]


def test_a_response_drawn_from_the_document_is_not_flagged():
    instruction = "Replace all nouns and verbs with their tags (e.g., [NOUN], [VERB])."
    assert has_instruction_vocabulary(["Barcelona hosted the summit last autumn."], [instruction]) == [False]


def test_a_response_with_no_word_characters_scores_zero():
    assert has_instruction_vocabulary(["___"], ["replace it with an underscore"]) == [False]
    assert has_instruction_vocabulary(["🌱🤝"], ["translate this into emojis"]) == [False]


def test_the_containment_threshold_is_configurable():
    assert has_instruction_vocabulary(["alpha beta"], ["alpha gamma"], threshold=0.4) == [True]
    assert has_instruction_vocabulary(["alpha beta"], ["alpha gamma"], threshold=0.6) == [False]


def test_an_echoed_instruction_line_is_flagged():
    instruction = "Output the full new text with no extra statements or commentations."
    assert has_prompt_echo([f"{BODY}\n{instruction}"], [instruction]) == [True]


def test_a_short_instruction_line_is_too_generic_to_match():
    assert has_prompt_echo(["Rewrite this."], ["Rewrite this."]) == [False]


def test_identical_and_near_identical_pairs_are_flagged():
    assert is_near_duplicate(["one two three four"], ["one two three four"]) == [True]
    assert is_near_duplicate(["one two three four"], ["five six seven eight"]) == [False]


def test_the_similarity_threshold_is_configurable():
    assert is_near_duplicate(["a b c d"], ["a b c e"], threshold=0.9) == [False]
    assert is_near_duplicate(["a b c d"], ["a b c e"], threshold=0.5) == [True]


def test_length_ratio_outliers_are_flagged_at_both_tails():
    original = "x" * 1000
    assert has_length_anomaly([original] * 3, ["x" * 50, "x" * 1000, "x" * 6000]) == [True, False, True]


def test_an_empty_original_is_a_length_anomaly():
    assert has_length_anomaly([""], ["anything"]) == [True]


def test_only_repeat_occurrences_are_flagged():
    assert is_duplicate(["a", "b", "a", "A", "c"]) == [False, False, True, True, False]


def test_duplicate_detection_ignores_case_and_whitespace():
    assert is_duplicate(["Cafe au lait", "cafe  au\nlait"]) == [False, True]


def test_scripts_that_would_transliterate_to_nothing_stay_distinct():
    # Folding accents away would map every emoji string onto the same key.
    assert is_duplicate(["🌱🤝", "🦅🏛️"]) == [False, False]


def test_group_keys_scope_collisions():
    texts = ["same", "same", "same"]
    assert is_duplicate(texts, groups=["a", "b", "a"]) == [False, False, True]


def test_scoping_does_not_change_the_ungrouped_result():
    texts = ["a", "b", "a"]
    assert is_duplicate(texts, groups=["g"] * 3) == is_duplicate(texts)
