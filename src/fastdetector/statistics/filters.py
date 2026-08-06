"""Row-level filters for the dataset defects catalogued in ISSUES.md.

Every function takes aligned lists and returns a list of the same length. Text
filters return cleaned strings; boolean filters return ``True`` for rows that
should be **removed**.

Issue coverage: 1 :func:`normalize_whitespace`, 2 :func:`strip_wrapper_boilerplate`,
3 :func:`has_filler_output` / :func:`has_instruction_vocabulary`,
4 :func:`has_refusal`, 5 :func:`is_near_duplicate`,
6 :func:`has_length_anomaly`, 7 :func:`is_empty`, 8 :func:`strip_added_title` /
:func:`strip_markdown` / :func:`strip_emoji`, 9 :func:`has_meta_commentary`,
10 :func:`has_prompt_echo`, 11 :func:`has_placeholder`, 13 :func:`is_duplicate`,
14 :func:`fix_encoding` / :func:`has_encoding_damage`. Issues 12 and 15 are prompt
and configuration defects; see PROMPT_FIXES.md.
"""

import re

import emoji
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

from fastdetector.statistics.statistics_basic import pairwise_jaccards

ACKNOWLEDGEMENT_PREFIXES = (
    "here ", "sure ", "here,", "sure,", "here:", "sure:", "sure!",
    "certainly ", "certainly,", "certainly!", "i'm happy to help ",
)

TITLE_PATTERN = re.compile(r"^\s*(?:#{1,6}[^\n]*|\*\*[^\n*]{1,120}\*\*:?)[ \t]*\n")

REFUSAL_PATTERN = re.compile(
    r"(?i)^\W*(?:i\s*(?:'m|am)?\s*(?:cannot|can't|can not|won't|will not|unable|not able)(?:\s+to)?\s+"
    r"(?:fulfill|complete|comply|assist|help|provide|create|generate|produce|write|proceed|continue|do)\b"
    r"|i'?m sorry\b|sorry, (?:but )?i\b|as an ai\b"
    r"|please provide (?:the|me)\b|you (?:haven't|have not) provided\b"
    r"|(?:it (?:seems|appears|looks like)|there (?:does\s?n.t appear to be|is no|are no)|unfortunately,)"
    r"[^.\n]{0,140}\b(?:input|document|descriptor|trace|passage|source text|provided|your message|no actual)\b)"
)

FILLER_PATTERN = re.compile(
    r"(?i)\b(?:the quick brown fox jumps over the lazy dog|lorem ipsum"
    r"|insert (?:the )?(?:source|original) text|source (?:text|material) "
    r"(?:to be |not )?provided)\b|\[(?:NOUN|VERB|ADJECTIVE|PUNCTUATION)\]"
)

PLACEHOLDER_PATTERN = re.compile(
    r"(?i)\[(?:insert|your|company|source|original|add|name|date|link|url|email|phone)"
    r"[^\]\n]{0,40}\]"
)

META_COMMENTARY_PATTERN = re.compile(
    r"(?i)\b(?:the (?:original|provided|source|above|following) "
    r"(?:text|passage|article|document)|here'?s? the (?:rewritten|revised|recreated|new)\b"
    r"|as (?:requested|instructed))"
)

MOJIBAKE_PATTERN = re.compile(r"Ã[\x80-\xbf]|â€|Â[\xa0-\xbf]")


def _as_strings(texts: list[str]) -> list[str]:
    """Coerce a column to strings, mapping missing values to the empty string.

    Args:
        texts: List of values to coerce.

    Returns:
        List of string values.
    """
    return [str(text) if text is not None else "" for text in texts]


def normalize_whitespace(texts: list[str]) -> list[str]:
    """Collapse whitespace so the human and AI columns share one convention.

    Issue 1. Source text arrives pre-normalized by the extractor while model
    output does not, so whitespace layout alone identifies the AI side with
    perfect precision. Apply this to *both* columns.

    Args:
        texts: List of text strings to normalize.

    Returns:
        List of texts with horizontal whitespace runs collapsed to one space,
        blank lines removed, and the edges stripped.
    """
    results = []
    for text in _as_strings(texts):
        results.append(re.sub(r"\s*\n\s*", "\n", re.sub(r"[^\S\n]+", " ", text)).strip())
    return results


def strip_wrapper_boilerplate(
    texts: list[str],
    originals: list[str],
    max_wrapper_words: int = 20,
    max_dropped_words: int = 40,
    max_dropped_proportion: float = 0.25,
) -> list[str]:
    """Remove an acknowledgement preamble and horizontal-rule wrapper.

    Issue 2. Drops a short conversational opening line, a short segment before
    the first ``---`` and a short segment after the last one, provided the
    original does not use those markers itself. Long segments are kept, so real
    trailing prose survives, and the result is always stripped.

    Args:
        texts: List of model responses to clean.
        originals: List of source texts aligned with ``texts``.
        max_wrapper_words: Longest segment still treated as a wrapper.
        max_dropped_words: Absolute word budget before a row is reverted.
        max_dropped_proportion: Relative word budget before a row is reverted.

    Returns:
        List of cleaned responses, reverted to the input where cleaning removed
        more than the allowed budget.
    """
    results = []
    for text, original in zip(_as_strings(texts), _as_strings(originals)):
        original_lower = original.strip().lower()
        cleaned = text.strip()

        if cleaned.lower().startswith(ACKNOWLEDGEMENT_PREFIXES) and not original_lower.startswith(ACKNOWLEDGEMENT_PREFIXES):
            opener, _, body = cleaned.partition("\n")
            if body.strip() and len(opener.split()) <= max_wrapper_words:
                cleaned = body.strip()

        if "---" not in original:
            opener, separator, body = cleaned.partition("---")
            if separator and len(opener.split()) <= max_wrapper_words:
                cleaned = body.strip()
            body, separator, trailer = cleaned.rpartition("---")
            if separator and len(trailer.split()) <= max_wrapper_words:
                cleaned = body.strip()

        word_count = len(text.split())
        dropped = word_count - len(cleaned.split())
        if dropped > max_dropped_words or (word_count and dropped / word_count > max_dropped_proportion):
            cleaned = text
        results.append(cleaned)
    return results


def strip_added_title(texts: list[str], originals: list[str]) -> list[str]:
    """Remove a leading heading or bold title line the original does not have.

    Issue 8. Covers both decorative titles and task labels such as
    ``**Recreated Text:**`` or ``**Rough Draft:**``, along with a horizontal
    rule left hanging underneath.

    Args:
        texts: List of model responses to clean.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of responses with an unmatched leading title line removed.
    """
    results = []
    for text, original in zip(_as_strings(texts), _as_strings(originals)):
        match = TITLE_PATTERN.match(text)
        if not match or TITLE_PATTERN.match(original):
            results.append(text)
            continue
        results.append(re.sub(r"^-{3,}[ \t]*\n", "", text[match.end():].lstrip()).lstrip())
    return results


def strip_markdown(texts: list[str]) -> list[str]:
    """Remove markdown constructs that models emit far more often than humans.

    Issue 8. Strips bold markers, heading markers, code fences and inline code,
    and reduces links to their anchor text. Bullet and numbered lists are left
    alone because both columns use them at the same rate. Apply to *both*
    columns.

    Args:
        texts: List of text strings to clean.

    Returns:
        List of texts with markdown markup removed.
    """
    results = []
    for text in _as_strings(texts):
        cleaned = re.sub(r"^\s*```[^\n]*\n?", "", text, flags=re.M).replace("`", "")
        cleaned = re.sub(r"\[([^\]\n]+)\]\([^)\n]*\)", r"\1", cleaned)
        cleaned = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", cleaned, flags=re.M)
        results.append(re.sub(r"\*\*|__", "", cleaned))
    return results


def strip_emoji(texts: list[str]) -> list[str]:
    """Remove emoji characters.

    Issue 8. Emoji injection is concentrated in a subset of the generators, so
    it is a model fingerprint rather than an AI signal. Apply to *both* columns.

    Args:
        texts: List of text strings to clean.

    Returns:
        List of texts with emoji removed.
    """
    return [emoji.replace_emoji(text, "") for text in _as_strings(texts)]


def fix_encoding(texts: list[str]) -> list[str]:
    """Repair mojibake and unescape literal escape sequences.

    Issue 14. Re-decodes text that was read as cp1252 but written as UTF-8, and
    converts literal ``\\n``/``\\t`` sequences into real whitespace.

    Args:
        texts: List of text strings to repair.

    Returns:
        List of repaired texts, left unchanged where re-decoding is not possible.
    """
    results = []
    for text in _as_strings(texts):
        repaired = text.replace("\\n", "\n").replace("\\t", "\t")
        if MOJIBAKE_PATTERN.search(repaired):
            try:
                repaired = repaired.encode("cp1252").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        results.append(repaired)
    return results


def is_empty(texts: list[str], min_characters: int = 50) -> list[bool]:
    """Flag responses that are empty or too short to carry any signal.

    Issue 7. A failed request returns an empty string, and the final turn's
    output is used unconditionally, so a failure on the last turn publishes a
    blank AI side.

    Args:
        texts: List of text strings to check.
        min_characters: Minimum stripped length a response must reach.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [len(text.strip()) < min_characters for text in _as_strings(texts)]


def has_refusal(texts: list[str]) -> list[bool]:
    """Flag responses that open with a refusal or a request for the source text.

    Issue 4. Only the opening is matched, so ordinary prose containing phrases
    such as "I will not let them win" is not flagged.

    Args:
        texts: List of text strings to check.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [bool(REFUSAL_PATTERN.match(text)) for text in _as_strings(texts)]


def has_filler_output(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses that fall back to pangrams, filler or part-of-speech tags.

    Issue 3. Produced when a reconstruction prompt receives a descriptor that
    carries no content from the source document. Sources that discuss filler
    text themselves are not flagged.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [
        bool(FILLER_PATTERN.search(text)) and not FILLER_PATTERN.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_placeholder(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses containing an unfilled bracketed template placeholder.

    Issue 11. Placeholders already present in the original are not flagged.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [
        bool(PLACEHOLDER_PATTERN.search(text)) and not PLACEHOLDER_PATTERN.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_meta_commentary(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses that narrate the rewriting task instead of performing it.

    Issue 9. Matches references to "the original text", "as requested" and
    similar task-level asides that the original does not contain.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [
        bool(META_COMMENTARY_PATTERN.search(text)) and not META_COMMENTARY_PATTERN.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_prompt_echo(texts: list[str], instructions: list[str], min_characters: int = 60) -> list[bool]:
    """Flag responses that reproduce a line of their own prompt verbatim.

    Issue 10. ``instructions`` holds the instruction text for each row with the
    document placeholder already removed, typically the prompt's chat turns
    joined by newlines.

    Args:
        texts: List of text strings to check.
        instructions: List of instruction texts aligned with ``texts``.
        min_characters: Shortest instruction line considered distinctive.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    results = []
    for text, instruction in zip(_as_strings(texts), _as_strings(instructions)):
        lines = [line.strip() for line in instruction.splitlines() if len(line.strip()) >= min_characters]
        results.append(any(line in text for line in lines))
    return results


def has_instruction_vocabulary(texts: list[str], instructions: list[str], threshold: float = 0.25) -> list[bool]:
    """Flag responses built from the instruction's words rather than the document's.

    Issue 3. Distinct from :func:`has_prompt_echo`, which looks for a verbatim
    span: a model that answers "replace nouns and verbs with tags such as
    ``[NOUN]``, ``[VERB]``" with ``[NOUN] [VERB] [ADJECTIVE] [CONJUNCTION]``
    copied nothing verbatim, it extrapolated the instruction's own example. The
    measure is therefore containment of the response's vocabulary in the
    instruction, not string overlap.

    Responses with no word characters at all score zero by definition; those are
    :func:`is_duplicate` territory.

    Args:
        texts: List of responses to check, typically an intermediate turn.
        instructions: List of instruction texts aligned with ``texts``, with the
            document placeholder already removed.
        threshold: Containment at or above which a response is flagged.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    results = []
    for text, instruction in zip(_as_strings(texts), _as_strings(instructions)):
        words = set(re.findall(r"\w+", text.lower()))
        shared = words & set(re.findall(r"\w+", instruction.lower()))
        results.append(bool(words) and len(shared) / len(words) >= threshold)
    return results


def is_near_duplicate(originals: list[str], texts: list[str], threshold: float = 0.9, n: int = 1) -> list[bool]:
    """Flag pairs whose two sides are too similar to carry a label signal.

    Issue 5. Similarity is word-level Jaccard by default; raise ``n`` to compare
    n-grams instead.

    Args:
        originals: List of source texts.
        texts: List of model responses aligned with ``originals``.
        threshold: Similarity at or above which a pair is flagged.
        n: Size of n-grams in words.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [1.0 - distance >= threshold for distance in pairwise_jaccards(originals, texts, n)]


def has_length_anomaly(
    originals: list[str],
    texts: list[str],
    min_ratio: float = 0.1,
    max_ratio: float = 5.0,
) -> list[bool]:
    """Flag pairs whose length ratio makes length a proxy for the label.

    Issue 6. Long sources starve the completion budget, so the AI side shrinks
    as the human side grows; the opposite tail is runaway expansion.

    Args:
        originals: List of source texts.
        texts: List of model responses aligned with ``originals``.
        min_ratio: Smallest acceptable response-to-original length ratio.
        max_ratio: Largest acceptable response-to-original length ratio.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    results = []
    for original, text in zip(_as_strings(originals), _as_strings(texts)):
        ratio = len(text) / len(original) if original else 0.0
        results.append(not min_ratio <= ratio <= max_ratio)
    return results


def is_duplicate(texts: list[str], groups: list[str] | None = None) -> list[bool]:
    """Flag every repeat occurrence of a text, keeping the first.

    Issue 13. Comparison ignores case and whitespace, but nothing else: folding
    accents away would also fold away emoji and non-Latin scripts, silently
    collapsing every such text onto the same key. Run this over the source
    column so that boilerplate such as cookie notices cannot straddle a
    train/test split.

    Passing ``groups`` scopes collisions to rows sharing a group key. Use it on
    an intermediate turn's responses, keyed by the instruction that produced
    them, to find encodings that cannot describe the document they came from:
    a descriptor identical to another document's descriptor encodes neither.
    Scoping is what makes that safe, because deliberately lossy instructions
    have a small output alphabet and collide with each other by design.

    Args:
        texts: List of text strings to check.
        groups: Optional list of group keys aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    seen = set()
    results = []
    for text, group in zip(_as_strings(texts), groups if groups is not None else texts):
        key = (group if groups is not None else None, " ".join(text.lower().split()))
        results.append(key in seen)
        seen.add(key)
    return results


def has_encoding_damage(texts: list[str]) -> list[bool]:
    """Flag texts still holding unrecoverable replacement characters.

    Issue 14. Run after :func:`fix_encoding`, which repairs what it can.

    Args:
        texts: List of text strings to check.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return ["�" in text for text in _as_strings(texts)]


def is_non_english(texts: list[str], threshold: float = 0.95) -> list[bool]:
    """Flag texts that are not confidently English.

    Args:
        texts: List of text strings to check.
        threshold: Minimum English probability a text must reach.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    results = []
    for text in _as_strings(texts):
        if not text.strip():
            results.append(True)
            continue
        try:
            results.append(not any(lang.lang == "en" and lang.prob >= threshold for lang in detect_langs(text)))
        except LangDetectException:
            results.append(True)
    return results
