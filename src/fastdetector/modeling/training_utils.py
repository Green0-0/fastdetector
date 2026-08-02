import os
import re

import emoji
import numpy as np
import optuna
from accelerate import PartialState
from datasets import load_dataset
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock


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


def editlens_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Discretize a cosine score into an ordinal bucket.

    Scores at or below ``lo`` collapse to bucket 0 and scores at or above ``hi``
    to the top bucket; the band between them is split evenly across the rest.
    Two buckets is the EditLens special case: there is no graded band to split,
    so the one boundary sits midway between ``lo`` and ``hi``.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Lower edge of the graded band.
        hi: Upper edge of the graded band.

    Returns:
        Bucket index in ``[0, n_buckets)``.
    """
    if n_buckets == 2:
        return 0 if score <= (lo + hi) / 2 else 1
    if score <= lo:
        return 0
    if score >= hi:
        return n_buckets - 1
    return 1 + int((score - lo) / (hi - lo) * (n_buckets - 2))


def naive_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Split the whole score range into buckets of equal width.

    Human text and lightly edited AI share the first bucket only because they
    share a score range, not by construction.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Unused; the band edges play no part here.
        hi: Unused; the band edges play no part here.

    Returns:
        Bucket index in ``[0, n_buckets)``.
    """
    return min(int(score * n_buckets), n_buckets - 1)


def human_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Reserve bucket 0 for human text, splitting the range over the rest.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Unused; the band edges play no part here.
        hi: Unused; the band edges play no part here.

    Returns:
        Bucket index in ``[0, n_buckets)``, 0 only for a score of 0.
    """
    if score <= 0:
        return 0
    return 1 + min(int(score * (n_buckets - 1)), n_buckets - 2)


def human_editlens_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Reserve bucket 0 for human text, banding the rest the EditLens way.

    The graded band is laid over the buckets above 0, so an AI row can no
    longer fall into the human bucket however lightly it was edited.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Lower edge of the graded band.
        hi: Upper edge of the graded band.

    Returns:
        Bucket index in ``[0, n_buckets)``, 0 only for a score of 0.
    """
    if score <= 0:
        return 0
    if n_buckets == 2:
        return 1
    return 1 + editlens_bucket(score, n_buckets - 1, lo, hi)


def human_size_adjusted_bucket(score: float, edges: np.ndarray) -> int:
    """Reserve bucket 0 for human text, giving the rest equal populations.

    Args:
        score: Cosine score to bucket.
        edges: Interior quantile edges of the AI scores, ascending, as built
            by :func:`make_bucketer`.

    Returns:
        Bucket index in ``[0, len(edges) + 2)``, 0 only for a score of 0.
    """
    if score <= 0:
        return 0
    return 1 + int(np.searchsorted(edges, score, side="right"))


BUCKETERS = {
    "editlens": editlens_bucket,
    "naive": naive_bucket,
    "human": human_bucket,
    "human_editlens": human_editlens_bucket,
}


def make_bucketer(strategy: str, n_buckets: int, lo: float, hi: float, scores: np.ndarray):
    """Build the score to bucket function a strategy names.

    Args:
        strategy: One of :data:`BUCKETERS` or ``"human_size"``.
        n_buckets: Total number of buckets.
        lo: Lower edge of the graded band.
        hi: Upper edge of the graded band.
        scores: Training scores, read only by ``human_size`` to place its
            quantile edges. Cutting them on training data alone keeps the
            eval split out of the labelling.

    Returns:
        Callable taking one score and returning its bucket index.

    Raises:
        KeyError: If the strategy is not one this module implements.
    """
    if strategy == "human_size":
        ai = scores[scores > 0]
        edges = (np.quantile(ai, np.linspace(0, 1, n_buckets)[1:-1])
                 if n_buckets > 2 and ai.size else np.empty(0))
        return lambda score: human_size_adjusted_bucket(score, edges)
    bucketer = BUCKETERS[strategy]
    return lambda score: bucketer(score, n_buckets, lo, hi)


def load_splits(dataset: str, tokenizer, bucket_strategy: str, n_buckets: int,
                lo: float, hi: float, max_length: int, min_words: int, seed: int) -> tuple:
    """Load, filter and tokenize the dataset's own train and val splits.

    Args:
        dataset: Dataset repository ID, carrying ``text`` and ``cosine_score``.
        tokenizer: Tokenizer matching the base model.
        bucket_strategy: Which labelling strategy to cut the scores with, as
            named in :func:`make_bucketer`.
        n_buckets: Number of discrete output bucket labels.
        lo: Lower edge of the graded scoring band.
        hi: Upper edge of the graded scoring band.
        max_length: Tokenizer truncation length.
        min_words: Shortest text to keep.
        seed: Seed the splits are shuffled with.

    Returns:
        Tuple of (train_dataset, val_dataset, val_scores), the datasets
        carrying input_ids, attention_mask and label, and val_scores holding
        the raw edit score the bucket label was rounded from.

        Both splits are kept whole. The val split used to drop the AI rows
        that the trial's own labelling binned as human, which is defensible
        when judging one labelling and ruinous when comparing several: the
        excluded rows are the lightly edited ones, so each labelling was
        marked against a different, self-selected test set. Measured on this
        corpus, a two bucket naive cut deleted 891 of the 894 edited texts
        while leaving every fully generated one, turning its score into
        "machine written text against human", which it answered at .998 and
        so led the study. Scoring every trial on the whole split lets a
        labelling that cannot separate a light edit score as badly as it
        deserves, which is the finding the ablation exists to surface.
    """
    def keep(row) -> bool:
        """Drop rows with no score, and rows too short to judge."""
        return row["cosine_score"] is not None and len(re.findall(r"\b\w+\b", (row["text"] or ""))) >= min_words

    def prep(batch) -> dict:
        """Clean and tokenize a batch, labelling it with its score's bucket."""
        encoded = tokenizer([clean_text(text) for text in batch["text"]],
                            truncation=True, max_length=max_length)
        encoded["label"] = [bucket(score) for score in batch["cosine_score"]]
        return encoded

    with PartialState().main_process_first():
        splits = [load_dataset(dataset, split=split).shuffle(seed=seed).filter(keep)
                  for split in ("train", "val")]
        bucket = make_bucketer(bucket_strategy, n_buckets, lo, hi,
                               np.array(splits[0]["cosine_score"], dtype=float))

        scores = np.array(splits[1]["cosine_score"], dtype=float)
        return (*(ds.map(prep, batched=True, remove_columns=ds.column_names)
                  for ds in splits), scores)


def get_or_create_study(study_name, journal_path, min_resource, reduction_factor,
                        max_resource="auto", direction="maximize"):
    """Open the study behind a journal file, creating it if it is not there.

    Args:
        study_name: Optuna study name, the key the journal is resumed under.
        journal_path: Path to the journal file; its directory is created.
        min_resource: Resource a trial runs through before Hyperband may prune
            it, in whatever unit the trials report.
        reduction_factor: Fraction of trials Hyperband keeps at each rung.
        max_resource: Resource a trial runs to. Pass the real budget rather
            than leaving it "auto": optuna infers "auto" from whichever trial
            happens to complete first, and the bracket count follows from
            max_resource / min_resource, so an unlucky first trial can quietly
            collapse the ladder to a single rung.
        direction: Direction to optimize the objective in.

    Returns:
        The optuna study.
    """
    directory = os.path.dirname(journal_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    lock = JournalFileOpenLock(f"{journal_path}.lock")
    storage = JournalStorage(JournalFileStorage(journal_path, lock_obj=lock))
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
