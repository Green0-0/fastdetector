import re
from accelerate import PartialState
from datasets import Dataset
import emoji


def clean_text(text: str) -> str:
    """Normalize text for EditLens inference and training.

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


def count_words(text: str) -> int:
    """Count word characters runs in a string.

    Args:
        text: Input text string.

    Returns:
        Number of words.
    """
    return len(re.findall(r"\b\w+\b", text))


def to_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Discretize a cosine score into an ordinal bucket.

    Scores at or below ``lo`` collapse to bucket 0 and scores at or above ``hi``
    to the top bucket; the band between them is split evenly across the rest.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Lower edge of the graded band.
        hi: Upper edge of the graded band.

    Returns:
        Bucket index in ``[0, n_buckets)``.
    """
    if score <= lo:
        return 0
    if score >= hi:
        return n_buckets - 1
    return 1 + int((score - lo) / (hi - lo) * (n_buckets - 2))


def _find_col(ds: Dataset, preferred: str, candidates: list[str]) -> str:
    """Resolve the actual dataset column name from preferred or candidate names.

    Args:
        ds: The HuggingFace Dataset.
        preferred: The primary expected column name.
        candidates: List of fallback candidate column names.

    Returns:
        The matched column name present in the dataset.

    Raises:
        ValueError: If neither preferred nor candidate column names exist.
    """
    if preferred in ds.column_names:
        return preferred
    for cand in candidates:
        if cand in ds.column_names:
            return cand
    raise ValueError(
        f"Could not find column '{preferred}' (candidates: {candidates}) in dataset: {ds.column_names}"
    )


def get_data(
    dataset: Dataset,
    tokenizer,
    n_buckets: int = 4,
    lo: float = 0.03,
    hi: float = 0.15,
    max_length: int = 1024,
    seed: int = 42,
    n_rows: int = -1,
    test_size: float = 0.05,
    human_col: str = "human",
    ai_col: str = "ai",
    edit_score_col: str = "edit_score",
    min_words: int = 75,
) -> tuple[Dataset, Dataset]:
    """Filter, discretize, and tokenize a dataset of human and AI text pairs.

    The dataset should have one column for human text, one for AI text, and one
    for edit scores. Human texts are assigned edit score = 0, while AI texts use
    their corresponding edit scores.

    Args:
        dataset: Input dataset containing human/AI texts and edit scores.
        tokenizer: Tokenizer matching the base model.
        n_buckets: Number of discrete output bucket labels.
        lo: Lower edge of score discretization band.
        hi: Upper edge of score discretization band.
        max_length: Maximum sequence length for tokenization.
        seed: Random seed for shuffling and splitting.
        n_rows: Maximum total samples to retain (-1 for all).
        test_size: Fraction of dataset reserved for evaluation split.
        human_col: Column name containing human text.
        ai_col: Column name containing AI text.
        edit_score_col: Column name containing edit scores.
        min_words: Minimum word count required for a text sample.

    Returns:
        Tuple of (train_dataset, eval_dataset), each carrying input_ids,
        attention_mask, and label.
    """
    h_col = _find_col(dataset, human_col, ["human", "human_text", "prompt"])
    a_col = _find_col(dataset, ai_col, ["ai", "ai_text", "response", "completion"])
    e_col = _find_col(dataset, edit_score_col, ["edit_score", "cosine_score", "score"])

    human_texts = dataset[h_col]
    ai_texts = dataset[a_col]
    scores = dataset[e_col]

    combined_texts = list(human_texts) + list(ai_texts)
    combined_scores = [0.0] * len(human_texts) + [
        float(s) if s is not None else 0.0 for s in scores
    ]

    ds = Dataset.from_dict({"text": combined_texts, "cosine_score": combined_scores})

    def keep(row):
        return (
            row["text"] is not None
            and row["cosine_score"] is not None
            and count_words(row["text"]) >= min_words
        )

    def prep(batch):
        encoded = tokenizer(
            [clean_text(text) for text in batch["text"]],
            truncation=True,
            max_length=max_length,
        )
        encoded["label"] = [
            to_bucket(score, n_buckets, lo, hi) for score in batch["cosine_score"]
        ]
        return encoded

    with PartialState().main_process_first():
        ds = ds.filter(keep).shuffle(seed=seed)
        if n_rows > 0:
            ds = ds.select(range(min(n_rows, len(ds))))
        ds = ds.map(prep, batched=True, remove_columns=ds.column_names)

    split = ds.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]