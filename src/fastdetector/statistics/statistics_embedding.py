"""Embedding-based text similarity metrics (cosine distance, BERTScore, MoverScore)."""

import math
import numpy as np
import torch
import ot
from collections import Counter

def pairwise_cosdist(embeddings_list_a: list[np.ndarray] | np.ndarray, embeddings_list_b: list[np.ndarray] | np.ndarray) -> list[float]:
    """Compute pairwise cosine distance between two aligned lists of normalized embeddings.

    Distance = 1 - cosine similarity, matching the "lower = more similar"
    convention of the other pairwise_* metrics.

    Args:
        embeddings_list_a: First list/array of normalized embeddings.
        embeddings_list_b: Second list/array of normalized embeddings.

    Returns:
        List of cosine distances.
    """
    embs_a = np.array(embeddings_list_a, dtype=np.float32)
    embs_b = np.array(embeddings_list_b, dtype=np.float32)
    cosdists = 1.0 - np.sum(embs_a * embs_b, axis=1)
    return cosdists.tolist()

def self_cossim_all(embeddings_list: list[np.ndarray] | np.ndarray, batch_size: int = 100) -> list[float]:
    """Compute the average cosine similarity of each embedding against all other embeddings in the same list.
    
    Args:
        embeddings_list: List/array of normalized embeddings.
        batch_size: Number of rows to process at once to prevent OOM errors.
        
    Returns:
        List of average cosine similarities.
    """
    embs = np.array(embeddings_list, dtype=np.float32)
    if len(embs) <= 1:
        # With 0 or 1 embeddings there are no "other" rows to average over.
        return [0.0] * len(embs)
    n_items = len(embs) - 1

    results = []
    for i in range(0, len(embs), batch_size):
        batch_embs = embs[i:i+batch_size]
        sims = batch_embs @ embs.T
        sum_sims = np.sum(sims, axis=1) - 1.0  # subtract self similarity
        results.extend((sum_sims / n_items).tolist())
    return results

def opposite_cossim_all(target_embeddings: list[np.ndarray] | np.ndarray, other_embeddings: list[np.ndarray] | np.ndarray, batch_size: int = 100) -> list[float]:
    """Compute the average cosine similarity of each target embedding against all other_embeddings.
    
    Args:
        target_embeddings: List/array of normalized target embeddings.
        other_embeddings: List/array of normalized reference embeddings.
        batch_size: Number of rows to process at once to prevent OOM errors.
        
    Returns:
        List of average cosine similarities.
    """
    target_embs = np.array(target_embeddings, dtype=np.float32)
    other_embs = np.array(other_embeddings, dtype=np.float32)
    if len(other_embs) == 0:
        return [0.0] * len(target_embs)
        
    results = []
    for i in range(0, len(target_embs), batch_size):
        batch_target = target_embs[i:i+batch_size]
        sims = batch_target @ other_embs.T
        results.extend(np.mean(sims, axis=1).tolist())
    return results

def _compute_idf(tokens_lists: list[list[str]]) -> dict[str, float]:
    """Compute inverse-document-frequency weights for a corpus of tokenized texts.

    Uses the standard smoothed IDF formula: ``log((N + 1) / (df + 1))``
    where N is the number of documents and df is the document frequency.

    Args:
        tokens_lists: List of token lists (one per document).

    Returns:
        Dict mapping each token to its IDF weight.
    """
    df = Counter()
    for tokens in tokens_lists:
        for token in set(tokens):
            df[token] += 1
    N = max(1, len(tokens_lists))
    return {token: math.log((N + 1) / (count + 1)) for token, count in df.items()}

def _get_idf_weights(tokens: list[str], idf_dict: dict[str, float]) -> np.ndarray:
    """Look up the IDF weight for each token in *tokens* and L1-normalize.

    Tokens not in *idf_dict* get weight 0. If all weights are near-zero
    (e.g. none of the tokens appear in the IDF dict), fall back to uniform
    weights so the caller doesn't divide by zero.

    Args:
        tokens: List of tokens to look up.
        idf_dict: Token -> IDF weight mapping (from :func:`_compute_idf`).

    Returns:
        Normalized weight vector (sums to 1), same length as *tokens*.
    """
    if not tokens:
        return np.array([])
    w = np.array([idf_dict.get(token, 0.0) for token in tokens])
    w = np.maximum(w, 0.0)
    if w.sum() <= 1e-12:
        w = np.ones(len(tokens))
    return w / w.sum()

def bertscore(src_embeddings_list: list[np.ndarray | torch.Tensor], edit_embeddings_list: list[np.ndarray | torch.Tensor], src_tokens_list: list[list[str]] | None = None, edit_tokens_list: list[list[str]] | None = None) -> tuple[list[float], list[float], list[float]]:
    """Compute BERTScore (Precision, Recall, F1) for aligned lists of token embedding matrices.
    
    Args:
        src_embeddings_list: List of token embeddings for reference texts (each N x D).
        edit_embeddings_list: List of token embeddings for candidate texts (each M x D).
        src_tokens_list: Optional list of tokenized reference texts for IDF term weighting.
        edit_tokens_list: Optional list of tokenized candidate texts for IDF term weighting.
        
    Returns:
        Tuple of (precisions, recalls, f1s), each a list of floats.
    """
    precisions = []
    recalls = []
    f1s = []
    
    src_idf = _compute_idf(src_tokens_list) if src_tokens_list is not None else None
    edit_idf = _compute_idf(edit_tokens_list) if edit_tokens_list is not None else None
    
    for i, (src, edit) in enumerate(zip(src_embeddings_list, edit_embeddings_list)):
        if isinstance(src, np.ndarray):
            src = torch.from_numpy(src)
        if isinstance(edit, np.ndarray):
            edit = torch.from_numpy(edit)
            
        if src.ndim != 2 or edit.ndim != 2 or src.shape[0] == 0 or edit.shape[0] == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
            
        sim_matrix = torch.mm(edit, src.t())  # M x NN
        
        if src_tokens_list is not None and edit_tokens_list is not None:
            w_src = _get_idf_weights(src_tokens_list[i], src_idf)
            w_edit = _get_idf_weights(edit_tokens_list[i], edit_idf)
            
            w_src_t = torch.from_numpy(w_src).to(src.device).float()
            w_edit_t = torch.from_numpy(w_edit).to(edit.device).float()
            
            precision = (sim_matrix.max(dim=1)[0] * w_edit_t).sum().item() if sim_matrix.size(0) > 0 else 0.0
            recall = (sim_matrix.max(dim=0)[0] * w_src_t).sum().item() if sim_matrix.size(1) > 0 else 0.0
        else:
            precision = sim_matrix.max(dim=1)[0].mean().item() if sim_matrix.size(0) > 0 else 0.0
            recall = sim_matrix.max(dim=0)[0].mean().item() if sim_matrix.size(1) > 0 else 0.0
            
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(1.0 - precision)
        recalls.append(1.0 - recall)
        f1s.append(1.0 - f1)
        
    return precisions, recalls, f1s

def moverscore(src_embeddings_list: list[np.ndarray | torch.Tensor], edit_embeddings_list: list[np.ndarray | torch.Tensor], src_tokens_list: list[list[str]] | None = None, edit_tokens_list: list[list[str]] | None = None) -> list[float]:
    """Compute MoverScore (Earth Mover's Distance) for aligned lists of token embedding matrices.
    
    Args:
        src_embeddings_list: List of token embeddings for reference texts (each N x D).
        edit_embeddings_list: List of token embeddings for candidate texts (each M x D).
        src_tokens_list: Optional list of tokenized reference texts for IDF term weighting.
        edit_tokens_list: Optional list of tokenized candidate texts for IDF term weighting.
        
    Returns:
        List of Earth Mover's Distances (lower is better, 0 means identical).
    """
    results = []
    
    src_idf = _compute_idf(src_tokens_list) if src_tokens_list is not None else None
    edit_idf = _compute_idf(edit_tokens_list) if edit_tokens_list is not None else None
    
    for i, (src, edit) in enumerate(zip(src_embeddings_list, edit_embeddings_list)):
        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        if isinstance(edit, torch.Tensor):
            edit = edit.detach().cpu().numpy()
            
        if src.ndim != 2 or edit.ndim != 2 or src.shape[0] == 0 or edit.shape[0] == 0:
            results.append(1.0)
            continue
            
        N, D = src.shape
        M, _ = edit.shape
            
        sim_matrix = src @ edit.T
        
        cost_matrix = np.sqrt(np.maximum(2.0 - 2.0 * sim_matrix, 0.0))
        
        if src_tokens_list is not None and edit_tokens_list is not None:
            weights_a = _get_idf_weights(src_tokens_list[i], src_idf)
            weights_b = _get_idf_weights(edit_tokens_list[i], edit_idf)
        else:
            weights_a = np.ones(N) / N
            weights_b = np.ones(M) / M
        
        distance = ot.emd2(weights_a, weights_b, cost_matrix)
        results.append(float(distance))
        
    return results

def pairwise_cosdist_chunked(a: list[np.ndarray], b: list[np.ndarray], operation: str = 'max') -> list[list[float]]:
    """Compute pairwise chunked cosine distance between two aligned lists of chunk embeddings.
    
    Args:
        a: First list of 2D arrays (num_chunks_a, D).
        b: Second list of 2D arrays (num_chunks_b, D).
        operation: 'mean', 'max', or 'min'.
        
    Returns:
        List of lists of floats.
    """
    results = []
    for embs_a, embs_b in zip(a, b):
        if embs_a is None or embs_b is None or embs_a.shape[0] == 0 or embs_b.shape[0] == 0:
            if embs_b is None or embs_b.shape[0] == 0:
                results.append([])
            else:
                results.append([1.0] * embs_b.shape[0])
            continue

        sims = embs_b @ embs_a.T
        dists = 1.0 - sims
        
        if operation == 'max':
            b_res = np.max(dists, axis=1)
        elif operation == 'min':
            b_res = np.min(dists, axis=1)
        elif operation == 'mean':
            b_res = np.mean(dists, axis=1)
        else:
            raise ValueError(f"Unrecognized operation: {operation}")
            
        results.append(b_res.tolist())
    return results