from fastdetector.statistics.statistics_basic import extract_ngrams
from typing import Dict
import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModel, AutoTokenizer

def _build_kwargs(model_name: str) -> Dict:
    """Returns the SentenceTransformer/CrossEncoder kwargs for a given model.

    Automatically applies flash attention, bf16, and left padding for Qwen3 models."""
    if "qwen3" in model_name.lower():
        return {
            "model_kwargs": {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
            "processor_kwargs": {"padding_side": "left"},
        }
    return {}


def _release_model(model) -> None:
    """Delete a model and reclaim CUDA memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def batch_gen_embeddings(
    texts: list[str],
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    batch_size: int = 4,
) -> np.ndarray:
    """Generate normalized embeddings for a list of texts.

    Args:
        texts: List of strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.

    Returns:
        Numpy array of normalized embeddings (shape: [len(texts), D]).
    """
    model = SentenceTransformer(model_name, trust_remote_code=True, **_build_kwargs(model_name))
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    _release_model(model)
    return embeddings


def batch_cross_encoder(
    texts_a: list[str],
    texts_b: list[str],
    model_name: str = "Qwen/Qwen3-Reranker-4B",
    batch_size: int = 2,
    as_distance: bool = True,
) -> list[float]:
    """Compute cross-encoder scores for aligned pairs of texts.

    Args:
        texts_a: First list of texts.
        texts_b: Second list of texts (same length as texts_a).
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        as_distance: If True, negate the scores so they behave like distances
            (lower = more similar), matching the convention of the other
            ``pairwise_*`` metrics.

    Returns:
        List of cross-encoder scores (negated if as_distance=True).
    """
    model = CrossEncoder(model_name, trust_remote_code=True, **_build_kwargs(model_name))
    pairs = list(zip(texts_a, texts_b))
    scores = model.predict(pairs, batch_size=batch_size)
    _release_model(model)

    if as_distance:
        return (-1.0 * scores).tolist()
    return scores.tolist()


def generate_token_embeddings_pairs(
    texts_a: list[str],
    texts_b: list[str],
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 4,
    chunk_size: int = 100,
):
    """Extract normalized token-level embeddings and subword tokens.

    Yields chunks of (embs_a, toks_a, embs_b, toks_b) to avoid OOM on large
    datasets. Each embs array is (num_tokens, D) with L2-normalized rows;
    special tokens ([CLS], [SEP]) are excluded.

    Args:
        texts_a: First list of texts.
        texts_b: Second list of texts.
        model_name: HuggingFace model identifier.
        batch_size: Inference batch size.
        chunk_size: How many texts to yield per iteration.

    Yields:
        Tuple of (embs_a, toks_a, embs_b, toks_b) for each chunk.
    """
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    def _extract_chunk(texts):
        all_embs = []
        all_tokens = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)

            for b_idx in range(len(batch_texts)):
                mask = inputs['attention_mask'][b_idx]
                length = mask.sum().item()
                if length > 2:
                    embs = outputs.last_hidden_state[b_idx, 1:length - 1]
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    all_embs.append(embs.cpu().numpy())
                    ids = inputs['input_ids'][b_idx, 1:length - 1]
                    all_tokens.append(tokenizer.convert_ids_to_tokens(ids))
                else:
                    all_embs.append(np.empty((0, outputs.last_hidden_state.size(-1))))
                    all_tokens.append([])
        return all_embs, all_tokens

    for i in range(0, len(texts_a), chunk_size):
        chunk_a = texts_a[i:i + chunk_size]
        chunk_b = texts_b[i:i + chunk_size]

        embs_a, toks_a = _extract_chunk(chunk_a)
        embs_b, toks_b = _extract_chunk(chunk_b)

        yield embs_a, toks_a, embs_b, toks_b

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _release_model(model)

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
    model = SentenceTransformer(model_name, trust_remote_code=True, **_build_kwargs(model_name))
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
