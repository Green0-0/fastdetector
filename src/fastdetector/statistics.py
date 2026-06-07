import math
import numpy as np
from collections import Counter
import Levenshtein
from typing import Optional

def global_ngram_analysis(texts: list[str], n: int) -> dict[str, float]:
    """Compute the global n-gram distribution across a list of texts.
    
    Args:
        texts: List of strings to analyze.
        n: The n-gram size.
        
    Returns:
        Dictionary mapping n-grams to their global frequencies.
    """
    counts = Counter()
    for text in texts:
        tokens = text.split()
        if len(tokens) >= n:
            counts.update([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total > 0 else {}

def ngram_analysis(texts: list[str], n: int) -> list[dict[str, float]]:
    """Compute the n-gram distribution for each text individually.
    
    Args:
        texts: List of strings to analyze.
        n: The n-gram size.
        
    Returns:
        List of dictionaries mapping n-grams to their frequencies for each text.
    """
    results = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            results.append({})
        else:
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            counts = Counter(ngrams)
            total = sum(counts.values())
            results.append({k: v / total for k, v in counts.items()})
    return results

def pairwise_jaccards(texts_list_a: list[str], texts_list_b: list[str], n: int) -> list[float]:
    """Compute pairwise Jaccard similarity between two aligned lists of texts using n-grams.
    
    Args:
        texts_list_a: First list of texts.
        texts_list_b: Second list of texts.
        n: The n-gram size.
        
    Returns:
        List of Jaccard similarity scores.
    """
    results = []
    for text1, text2 in zip(texts_list_a, texts_list_b):
        t1_tokens = text1.split()
        t2_tokens = text2.split()
        t1 = set([" ".join(t1_tokens[i:i+n]) for i in range(len(t1_tokens) - n + 1)])
        t2 = set([" ".join(t2_tokens[i:i+n]) for i in range(len(t2_tokens) - n + 1)])
        
        if not t1 and not t2:
            results.append(1.0 if text1.strip() == text2.strip() else 0.0)
        elif not t1 or not t2:
            results.append(0.0)
        else:
            results.append(len(t1.intersection(t2)) / len(t1.union(t2)))
    return results

def pairwise_levenshteins(texts_list_a: list[str], texts_list_b: list[str]) -> list[float]:
    """Compute pairwise Levenshtein distance between two aligned lists of texts.
    
    Args:
        texts_list_a: First list of texts.
        texts_list_b: Second list of texts.
        
    Returns:
        List of Levenshtein distances (as floats).
    """
    return [float(Levenshtein.distance(t1, t2)) for t1, t2 in zip(texts_list_a, texts_list_b)]

def entropies_approx(texts: list[str], top_logprobs: list[list[dict[str, float]]]) -> list[float]:
    """Approximate mean next-token entropy for each text using top-N logprobs and a tail-mass heuristic.
    
    Args:
        texts: List of strings.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        
    Returns:
        List of approximated entropy values.
    """
    results = []
    for text, text_top_lps in zip(texts, top_logprobs):
        if not text.strip() or not text_top_lps:
            results.append(0.0)
            continue
            
        entropies = []
        for top_lps in text_top_lps:
            if top_lps is not None and len(top_lps) > 0:
                p = np.array([math.exp(lp) for lp in top_lps.values()])
                h_top = -np.sum(p * np.log(p + 1e-12))
                
                Z = np.sum(p)
                M = max(0.0, 1.0 - Z)
                
                if M > 0:
                    p_min = np.min(p)
                    p_bound = min(p_min, M)
                    h_tail = -M * math.log(p_bound + 1e-12)
                else:
                    h_tail = 0.0
                entropies.append(h_top + h_tail)
                
        results.append(float(np.mean(entropies)) if entropies else 0.0)
    return results

def perplexities(texts: list[str], token_logprobs: list[list[Optional[float]]]) -> list[float]:
    """Compute perplexity for each text.
    
    Args:
        texts: List of strings.
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        
    Returns:
        List of perplexity values.
    """
    results = []
    for text, text_token_lps in zip(texts, token_logprobs):
        if not text.strip() or not text_token_lps:
            results.append(0.0)
            continue
            
        valid_lps = [lp for lp in text_token_lps if lp is not None]
        if not valid_lps:
            results.append(0.0)
            continue
            
        avg_logprob = sum(valid_lps) / len(valid_lps)
        try:
            results.append(math.exp(-avg_logprob))
        except OverflowError:
            results.append(float('inf'))
    return results

def top_p_outlier_percentages(texts: list[str], top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[Optional[float]]], p: float) -> list[float]:
    """Compute the percentage of tokens outside the top-p probability mass for each text.
    
    Args:
        texts: List of strings.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to logprobs.
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        p: Probability mass threshold (e.g., 0.9).
        
    Returns:
        List of outlier percentages.
    """
    results = []
    for text, text_top_lps, text_token_lps in zip(texts, top_logprobs, token_logprobs):
        if not text.strip() or not text_top_lps or not text_token_lps:
            results.append(0.0)
            continue
            
        outlier_count = 0
        total_count = 0
        
        for token_lp, top_lps in zip(text_token_lps, text_top_lps):
            if token_lp is not None and top_lps is not None and len(top_lps) > 0:
                total_count += 1
                sorted_probs = sorted([math.exp(lp) for lp in top_lps.values()], reverse=True)
                cumulative = 0.0
                threshold_prob = 0.0
                for prob in sorted_probs:
                    cumulative += prob
                    if cumulative >= p:
                        threshold_prob = prob
                        break
                if cumulative < p:
                    threshold_prob = sorted_probs[-1]
                    
                token_prob = math.exp(token_lp)
                if token_prob < threshold_prob - 1e-6:
                    outlier_count += 1
                    
        results.append(outlier_count / total_count if total_count > 0 else 0.0)
    return results

def top_k_outlier_percentages(texts: list[str], top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[Optional[float]]], k: int) -> list[float]:
    """Compute the percentage of tokens outside the top-k probability mass for each text.
    
    Args:
        texts: List of strings.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to logprobs.
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        k: Top-k threshold.
        
    Returns:
        List of outlier percentages.
    """
    results = []
    for text, text_top_lps, text_token_lps in zip(texts, top_logprobs, token_logprobs):
        if not text.strip() or not text_top_lps or not text_token_lps:
            results.append(0.0)
            continue
            
        outlier_count = 0
        total_count = 0
        
        for token_lp, top_lps in zip(text_token_lps, text_top_lps):
            if token_lp is None or top_lps is None or len(top_lps) < k:
                continue
                
            total_count += 1
            sorted_lps = sorted(top_lps.values(), reverse=True)
            kth_lp = sorted_lps[k-1]
            if token_lp < kth_lp - 1e-5:
                outlier_count += 1
                
        results.append(outlier_count / total_count if total_count > 0 else 0.0)
    return results
    
def pairwise_cossim(embeddings_list_a: list[np.ndarray] | np.ndarray, embeddings_list_b: list[np.ndarray] | np.ndarray) -> list[float]:
    """Compute pairwise cosine similarity between two aligned lists of embeddings.
    
    Args:
        embeddings_list_a: First list/array of normalized embeddings.
        embeddings_list_b: Second list/array of normalized embeddings.
        
    Returns:
        List of cosine similarities.
    """
    embs_a = np.array(embeddings_list_a, dtype=np.float32)
    embs_b = np.array(embeddings_list_b, dtype=np.float32)
    cossims = np.sum(embs_a * embs_b, axis=1)
    return cossims.tolist()

def self_cossim_all(embeddings_list: list[np.ndarray] | np.ndarray, batch_size: int = 100) -> list[float]:
    """Compute the average cosine similarity of each embedding against all other embeddings in the same list.
    
    Args:
        embeddings_list: List/array of normalized embeddings.
        batch_size: Number of rows to process at once to prevent OOM errors.
        
    Returns:
        List of average cosine similarities.
    """
    embs = np.array(embeddings_list, dtype=np.float32)
    n_items = max(1, len(embs) - 1)
    if n_items == 0:
        return [0.0] * len(embs)
        
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

def quantile(values: list[float]) -> list[float]:
    """Compute the quantile (percentile rank in decimal) for each value in a list of floats.
    Uses average rank for ties.
    
    Args:
        values: List of floats.
        
    Returns:
        List of floats between 0.0 and 1.0.
    """
    if not values:
        return []
    n = len(values)
    if n == 1:
        return [1.0]
    arr = np.array(values, dtype=float)
    sorted_arr = np.sort(arr)
    
    left_ranks = np.searchsorted(sorted_arr, arr, side='left')
    right_ranks = np.searchsorted(sorted_arr, arr, side='right')
    
    avg_ranks = (left_ranks + 1 + right_ranks) / 2.0
    return (avg_ranks / n).tolist()

def min_max_norm(values: list[float]) -> list[float]:
    """Compute the min-max normalization for a list of floats.
    
    Args:
        values: List of floats.
        
    Returns:
        List of floats scaled between 0.0 and 1.0.
    """
    if not values:
        return []
    n = len(values)
    if n == 1:
        return [0.0]
    arr = np.array(values, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if min_val == max_val:
        return [0.0] * n
    return ((arr - min_val) / (max_val - min_val)).tolist()