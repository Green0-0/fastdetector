import re
import emoji

def clean_text(text: str) -> str:
    """Normalize text for EditLens inference.

    Args:
        text: Input text string.

    Returns:
        Normalized text string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = emoji.demojize(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text