"""Soft n-gram distance scoring via phrase-level embedding similarity.

Split out of statistics_api.py for modularity.

Soft n-gram distance is a fuzzy alternative to Jaccard distance: instead of
requiring exact n-gram matches, it counts an n-gram in the edited text as
"matched" if any n-gram in the source text has cosine similarity >= threshold
under a sentence embedding model.
"""

import gc

import torch
from sentence_transformers import SentenceTransformer

from fastdetector.statistics.statistics_basic import extract_ngrams
from fastdetector.statistics.embeddings_api import _build_sentence_transformer, _release_model


def batch_soft_ngram_scores(
    source_texts: list[str],
    edited_texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.8,
    min_length: int = 6,
    max_length: int = 12,
    phrase_batch_size: int = 2048,
) -> list[float]:
    """Compute soft n-gram distance between pairs of source and edited texts.

    For each (source, edited) pair:
    1. Extract all word n-grams (lengths min_length..max_length) from both texts.
    2. Embed all unique n-grams across the current chunk using a sentence
       embedding model.
    3. For each n-gram in the edited text, check if any n-gram in the source
       text has cosine similarity >= threshold. If so, count it as matched.
    4. Distance = 1 - precision = 1 - (matched / total edited n-grams).

    Processing is done in chunks of `doc_batch_size=100` documents to prevent
    CUDA OOM on massive datasets.

    Args:
        source_texts: List of original texts.
        edited_texts: List of edited/generated texts (same length as source_texts).
        model_name: HuggingFace sentence-transformers model ID.
        threshold: Cosine similarity threshold for counting a phrase as matched.
        min_length: Minimum n-gram length in words.
        max_length: Maximum n-gram length in words.
        phrase_batch_size: Encoding batch size for soft n-gram phrases.

    Returns:
        List of distance scores (1 - precision). Higher means more dissimilar.
    """
    model = _build_sentence_transformer(model_name)
    results = []

    doc_batch_size = 100
    for i in range(0, len(source_texts), doc_batch_size):
        chunk_src = source_texts[i:i + doc_batch_size]
        chunk_edit = edited_texts[i:i + doc_batch_size]

        src_phrases_list = extract_ngrams(chunk_src, min_length=min_length, max_length=max_length)
        edit_phrases_list = extract_ngrams(chunk_edit, min_length=min_length, max_length=max_length)

        unique_phrases = list(set(
            phrase
            for phrases in src_phrases_list + edit_phrases_list
            for phrase in phrases
        ))

        if not unique_phrases:
            results.extend([1.0] * len(chunk_src))
            continue

        print(
            f"Batch {i // doc_batch_size + 1}/{(len(source_texts) + doc_batch_size - 1) // doc_batch_size}: "
            f"Encoding {len(unique_phrases)} unique phrases...",
            flush=True,
        )

        all_embeddings = model.encode(
            unique_phrases,
            convert_to_tensor=True,
            batch_size=phrase_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        phrase_to_idx = {phrase: idx for idx, phrase in enumerate(unique_phrases)}

        for src_phrases, edit_phrases in zip(src_phrases_list, edit_phrases_list):
            if not src_phrases or not edit_phrases:
                results.append(1.0)
                continue

            src_indices = [phrase_to_idx[p] for p in src_phrases]
            edit_indices = [phrase_to_idx[p] for p in edit_phrases]

            src_embeddings = all_embeddings[src_indices]
            edit_embeddings = all_embeddings[edit_indices]

            similarity_matrix = torch.mm(src_embeddings, edit_embeddings.t())

            matches = (similarity_matrix >= threshold).any(dim=0)
            precision = matches.sum().item() / len(edit_phrases)

            results.append(1.0 - precision)

        del all_embeddings
        try:
            del similarity_matrix
        except NameError:
            pass

    _release_model(model)
    return results
