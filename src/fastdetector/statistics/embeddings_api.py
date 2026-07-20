"""Sentence-level and token-level embedding generation via HuggingFace models.

Split out of statistics_api.py for modularity.

- batch_gen_embeddings: sentence-level embeddings via SentenceTransformer
- generate_token_embeddings_pairs: token-level embeddings via AutoModel,
  yielded in chunks to avoid OOM
- batch_cross_encoder: cross-encoder (reranker) scores for aligned text pairs
"""

import gc

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModel, AutoTokenizer


def _qwen3_kwargs() -> dict:
    """Return SentenceTransformer/CrossEncoder kwargs for Qwen3 models.

    Qwen3 models require flash_attention_2, bfloat16, and left padding.
    This is applied when the model name contains 'qwen3'.
    """
    return {
        "model_kwargs": {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        "processor_kwargs": {"padding_side": "left"},
    }


def _build_sentence_transformer(model_name: str) -> SentenceTransformer:
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs = _qwen3_kwargs()
    return SentenceTransformer(model_name, trust_remote_code=True, **kwargs)


def _build_cross_encoder(model_name: str) -> CrossEncoder:
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs = _qwen3_kwargs()
    return CrossEncoder(model_name, trust_remote_code=True, **kwargs)


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
    model = _build_sentence_transformer(model_name)
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
            (lower = more similar). This is the default for compatibility with
            distance-based metrics downstream.

    Returns:
        List of cross-encoder scores (negated if as_distance=True).
    """
    model = _build_cross_encoder(model_name)
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
