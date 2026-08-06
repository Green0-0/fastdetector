import re

import emoji
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException

from fastdetector.statistics.statistics_basic import pairwise_jaccards


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
    acknowledgement_prefixes = (
        "here ", "sure ", "here,", "sure,", "here:", "sure:", "sure!",
        "certainly ", "certainly,", "certainly!", "i'm happy to help ",
    )
    results = []
    for text, original in zip(_as_strings(texts), _as_strings(originals)):
        original_lower = original.strip().lower()
        cleaned = text.strip()

        if cleaned.lower().startswith(acknowledgement_prefixes) and not original_lower.startswith(acknowledgement_prefixes):
            opener, _, body = cleaned.partition("\n")
            if body.strip() and len(opener.split()) <= max_wrapper_words:
                cleaned = body.strip()

        RULE_LINE = re.compile(r"(?m)^[ \t]*-{3,}[ \t]*$")

        if not RULE_LINE.search(original):
            rules = list(RULE_LINE.finditer(cleaned))
            if rules and len(cleaned[:rules[0].start()].split()) <= max_wrapper_words:
                cleaned = cleaned[rules[0].end():].strip()
                rules = list(RULE_LINE.finditer(cleaned))
            if len(rules) == 1 and len(cleaned[rules[0].end():].split()) <= max_wrapper_words:
                cleaned = cleaned[:rules[0].start()].strip()

        word_count = len(text.split())
        dropped = word_count - len(cleaned.split())
        if dropped > max_dropped_words or (word_count and dropped / word_count > max_dropped_proportion):
            cleaned = text
        results.append(cleaned)
    return results


def strip_added_title(texts: list[str], originals: list[str]) -> list[str]:
    """Remove a leading heading or bold title line the original does not have.

    Args:
        texts: List of model responses to clean.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of responses with an unmatched leading title line removed.
    """
    title_pattern = re.compile(r"^\s*(?:#{1,6}[^\n]*|\*\*[^\n*]{1,120}\*\*:?)[ \t]*\n")
    results = []
    for text, original in zip(_as_strings(texts), _as_strings(originals)):
        match = title_pattern.match(text)
        if not match or title_pattern.match(original):
            results.append(text)
            continue
        results.append(re.sub(r"^-{3,}[ \t]*\n", "", text[match.end():].lstrip()).lstrip())
    return results


def strip_markdown(texts: list[str]) -> list[str]:
    """Remove markdown constructs that models emit far more often than humans.

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

    Args:
        texts: List of text strings to clean.

    Returns:
        List of texts with emoji removed.
    """
    return [emoji.replace_emoji(text, "") for text in _as_strings(texts)]


def fix_encoding(texts: list[str]) -> list[str]:
    """Repair mojibake and unescape literal escape sequences.

    Args:
        texts: List of text strings to repair.

    Returns:
        List of repaired texts, left unchanged where re-decoding is not possible.
    """
    mojibake_pattern = re.compile(r"Ã[\x80-\xbf]|â€|Â[\xa0-\xbf]")
    results = []
    for text in _as_strings(texts):
        repaired = text.replace("\\n", "\n").replace("\\t", "\t")
        if mojibake_pattern.search(repaired):
            try:
                repaired = repaired.encode("cp1252").decode("utf-8")
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        results.append(repaired)
    return results


def is_empty(texts: list[str], min_characters: int = 50) -> list[bool]:
    """Flag responses that are empty or too short to carry any signal.

    Args:
        texts: List of text strings to check.
        min_characters: Minimum stripped length a response must reach.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [len(text.strip()) < min_characters for text in _as_strings(texts)]


def has_refusal(texts: list[str]) -> list[bool]:
    """Flag responses that open with a refusal or a request for the source text.

    Args:
        texts: List of text strings to check.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    refusal_pattern = re.compile(
        r"(?i)^\W*(?:i\s*(?:'m|am)?\s*(?:cannot|can't|can not|won't|will not|unable|not able)(?:\s+to)?\s+"
        r"(?:fulfill|complete|comply|assist|help|provide|create|generate|produce|write|proceed|continue|do)\b"
        r"|i'?m sorry\b|sorry, (?:but )?i\b|as an ai\b"
        r"|please provide (?:the|me)\b|you (?:haven't|have not) provided\b"
        r"|(?:it (?:seems|appears|looks like)|there (?:does\s?n.t appear to be|is no|are no)|unfortunately,)"
        r"[^.\n]{0,140}\b(?:input|document|descriptor|trace|passage|source text|provided|your message|no actual)\b)"
    )
    return [bool(refusal_pattern.match(text)) for text in _as_strings(texts)]


def has_filler_output(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses that fall back to pangrams, filler or part-of-speech tags.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    filler_pattern = re.compile(
        r"(?i)\b(?:the quick brown fox jumps over the lazy dog|lorem ipsum"
        r"|insert (?:the )?(?:source|original) text|source (?:text|material) "
        r"(?:to be |not )?provided)\b|\[(?:NOUN|VERB|ADJECTIVE|PUNCTUATION)\]"
    )
    return [
        bool(filler_pattern.search(text)) and not filler_pattern.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_placeholder(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses containing an unfilled bracketed template placeholder.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    placeholder_pattern = re.compile(
        r"(?i)\[(?:insert|add|enter|your|their|client|company|recipient|author|product)\b[^\]\n]{0,40}\]"
        r"|\[(?:name|date|title|address|city|phone number|email address|link|url|website)\]"
        r"|\[(?:source text|original (?:text|passage)|placeholder)[^\]\n]{0,30}\]"
    )
    return [
        bool(placeholder_pattern.search(text)) and not placeholder_pattern.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def is_unchanged(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses the model handed back verbatim instead of rewriting.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return [
        text == original
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_meta_commentary(texts: list[str], originals: list[str]) -> list[bool]:
    """Flag responses that narrate the rewriting task instead of performing it.

    Args:
        texts: List of text strings to check.
        originals: List of source texts aligned with ``texts``.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    meta_commentary_pattern = re.compile(
        r"(?i)\bthe (?:original|provided|source|above) (?:text|passage|article|document)\b"
        r"|\b(?:here'?s|here is|below is) the (?:rewritten|revised|recreated|reworded|new|updated) "
        r"(?:text|version|passage|draft)\b"
        r"|\bi (?:have |'ve )?(?:rewritten|revised|recreated|reworded) (?:the|this|it)\b"
    )
    return [
        bool(meta_commentary_pattern.search(text)) and not meta_commentary_pattern.search(original)
        for text, original in zip(_as_strings(texts), _as_strings(originals))
    ]


def has_prompt_echo(texts: list[str], instructions: list[str], min_characters: int = 60) -> list[bool]:
    """Flag responses that reproduce a line of their own prompt verbatim.

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

    Args:
        texts: List of text strings to check.

    Returns:
        List of booleans, ``True`` where the row should be removed.
    """
    return ["\ufffd" in text for text in _as_strings(texts)]


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