"""Basic text-similarity metrics (n-grams, Jaccard, Levenshtein, subset checks, quantiles)."""

import numpy as np
from collections import Counter
import Levenshtein
from unidecode import unidecode

def global_ngram_analysis(texts: list[str], n: int) -> dict[str, int]:
    """Compute the global n-gram distribution across a list of texts.
    
    Args:
        texts: List of strings to analyze.
        n: The n-gram size.
        
    Returns:
        Dictionary mapping n-grams to their raw frequencies.
    """
    counts = Counter()
    for text in texts:
        tokens = text.split()
        if len(tokens) >= n:
            counts.update([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    return dict(counts)

def ngram_analysis(texts: list[str], n: int) -> list[dict[str, int]]:
    """Compute the n-gram distribution for each text individually.
    
    Args:
        texts: List of strings to analyze.
        n: The n-gram size.
        
    Returns:
        List of dictionaries mapping n-grams to their raw counts for each text.
    """
    results = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            results.append({})
        else:
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            counts = Counter(ngrams)
            results.append(dict(counts))
    return results

def extract_ngrams(texts: list[str], min_length: int = 6, max_length: int = 12) -> list[list[str]]:
    """Extract all contiguous word n-grams of lengths between min_length and max_length for each text.
    
    Args:
        texts: List of input texts.
        min_length: Minimum n-gram length in words.
        max_length: Maximum n-gram length in words.
        
    Returns:
        List of lists, where each inner list contains the n-gram phrases for the corresponding text.
    """
    results = []
    for text in texts:
        words = text.split()
        if not words:
            results.append([])
            continue
            
        current_min_length = min_length
        if len(words) < current_min_length:
            current_min_length = len(words)
            
        phrases = []
        for length in range(current_min_length, max_length + 1):
            for i in range(len(words) - length + 1):
                phrases.append(" ".join(words[i : i + length]))
        results.append(phrases)
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
            results.append(0.0 if text1.strip() == text2.strip() else 1.0)
        elif not t1 or not t2:
            results.append(1.0)
        else:
            results.append(1.0 - (len(t1.intersection(t2)) / len(t1.union(t2))))
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

def deviated_lines(texts_a: list[str], texts_b: list[str]) -> tuple[list[float], list[int]]:
    """Compute the proportion and raw count of deviated lines between pairs of texts."""
    proportions = []
    raw_counts = []
    for a, b in zip(texts_a, texts_b):
        a_str = str(a) if a is not None else ""
        b_str = str(b) if b is not None else ""
        a_norm = unidecode(a_str)
        b_norm = unidecode(b_str)
        
        a_lines = len(a_norm.splitlines()) if a_norm else 0
        b_lines = len(b_norm.splitlines()) if b_norm else 0
        dl = abs(a_lines - b_lines)
        raw_counts.append(dl)
        max_lines = max(a_lines, b_lines)
        proportions.append(dl / max_lines if max_lines > 0 else 0.0)
    return proportions, raw_counts

def deviated_words(texts_a: list[str], texts_b: list[str]) -> tuple[list[float], list[int]]:
    """Compute the proportion and raw count of deviated words between pairs of texts."""
    proportions = []
    raw_counts = []
    for a, b in zip(texts_a, texts_b):
        a_str = str(a) if a is not None else ""
        b_str = str(b) if b is not None else ""
        a_norm = unidecode(a_str)
        b_norm = unidecode(b_str)
        
        a_words = len(a_norm.split())
        b_words = len(b_norm.split())
        dw = abs(a_words - b_words)
        raw_counts.append(dw)
        max_words = max(a_words, b_words)
        proportions.append(dw / max_words if max_words > 0 else 0.0)
    return proportions, raw_counts

def deviated_characters(texts_a: list[str], texts_b: list[str]) -> tuple[list[float], list[int]]:
    """Compute the proportion and raw count of deviated characters between pairs of texts."""
    proportions = []
    raw_counts = []
    for a, b in zip(texts_a, texts_b):
        a_str = str(a) if a is not None else ""
        b_str = str(b) if b is not None else ""
        a_norm = unidecode(a_str)
        b_norm = unidecode(b_str)
        
        a_chars = len(a_norm)
        b_chars = len(b_norm)
        dc = abs(a_chars - b_chars)
        raw_counts.append(dc)
        max_chars = max(a_chars, b_chars)
        proportions.append(dc / max_chars if max_chars > 0 else 0.0)
    return proportions, raw_counts

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

def is_strict_subset(texts_a: list[str], texts_b: list[str]) -> list[bool]:
    """Check if texts_b are strictly substrings of texts_a."""
    results = []
    for a, b in zip(texts_a, texts_b):
        a_str = str(a) if a is not None else ""
        b_str = str(b) if b is not None else ""
        if not b_str or b_str not in a_str:
            results.append(False)
        else:
            results.append(True)
    return results

def is_loose_subset(texts_a: list[str], texts_b: list[str]) -> tuple[list[bool], list[str]]:
    """Check if texts_b are loose substrings of texts_a, ignoring spaces, case, and unicode differences.
    Returns a tuple of (is_subset, collected_subset)."""
    is_subsets = []
    collected_subsets = []
    for a, b in zip(texts_a, texts_b):
        a_str = str(a) if a is not None else ""
        b_str = str(b) if b is not None else ""
        
        b_norm = unidecode(b_str)
        
        orig_canon_parts = []
        orig_mapping = []
        for i, c in enumerate(a_str):
            norm_c = unidecode(c)
            for nc in norm_c:
                if not nc.isspace():
                    lowered = nc.lower()
                    orig_canon_parts.append(lowered)
                    orig_mapping.extend([i] * len(lowered))
        orig_canon = "".join(orig_canon_parts)
        
        new_canon = "".join(c.lower() for c in b_norm if not c.isspace())
        
        if not new_canon or new_canon not in orig_canon:
            is_subsets.append(False)
            collected_subsets.append("")
        else:
            is_subsets.append(True)
            start_idx = orig_canon.find(new_canon)
            end_idx = start_idx + len(new_canon) - 1
            orig_start = orig_mapping[start_idx]
            orig_end = orig_mapping[end_idx]
            collected_subsets.append(a_str[orig_start:orig_end+1])
            
    return is_subsets, collected_subsets

def sliding_window_word_chunk(texts: list[str], window_size: int, step_size: int) -> list[list[str]]:
    """Chunks each string into a list of strings using a sliding window.
    
    Args:
        texts: List of strings.
        window_size: Number of words in each chunk.
        step_size: Number of words to slide the window.
        
    Returns:
        List of lists of strings, where each inner list contains chunks for the corresponding text.
    """
    results = []
    for text in texts:
        if text is None:
            results.append([])
            continue
        words = text.split()
        if not words:
            results.append([])
            continue
            
        chunks = []
        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + window_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
        results.append(chunks)
    return results

def pairwise_chunked_jaccards(a: list[list[str]], b: list[list[str]], n: int, operation: str = 'max') -> list[list[float]]:
    """Compute pairwise chunked Jaccard distance between two aligned lists of chunk lists.
    
    Args:
        a: First list of chunk lists.
        b: Second list of chunk lists.
        n: The n-gram size.
        operation: 'mean', 'max', or 'min'.
        
    Returns:
        List of lists of floats.
    """
    results = []
    for chunks_a, chunks_b in zip(a, b):
        if not chunks_a or not chunks_b:
            if not chunks_b:
                results.append([])
            else:
                results.append([1.0] * len(chunks_b))
            continue
            
        a_sets = []
        for chunk in chunks_a:
            tokens = chunk.split()
            a_sets.append(set([" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]))
            
        b_res = []
        for chunk_b in chunks_b:
            tokens_b = chunk_b.split()
            b_set = set([" ".join(tokens_b[i:i+n]) for i in range(len(tokens_b) - n + 1)])
            
            chunk_scores = []
            for chunk_a, a_set in zip(chunks_a, a_sets):
                if not a_set and not b_set:
                    chunk_scores.append(0.0 if chunk_a.strip() == chunk_b.strip() else 1.0)
                elif not a_set or not b_set:
                    chunk_scores.append(1.0)
                else:
                    chunk_scores.append(1.0 - (len(a_set.intersection(b_set)) / len(a_set.union(b_set))))
            
            if operation == 'max':
                b_res.append(max(chunk_scores))
            elif operation == 'min':
                b_res.append(min(chunk_scores))
            elif operation == 'mean':
                b_res.append(sum(chunk_scores) / len(chunk_scores))
            else:
                raise ValueError(f"Unrecognized operation: {operation}")
        results.append(b_res)
    return results

def pairwise_chunked_levenshteins(a: list[list[str]], b: list[list[str]], operation: str = 'max') -> list[list[float]]:
    """Compute pairwise chunked Levenshtein distance between two aligned lists of chunk lists."""
    results = []
    for chunks_a, chunks_b in zip(a, b):
        if not chunks_a or not chunks_b:
            if not chunks_b:
                results.append([])
            else:
                results.append([float(len(cb)) for cb in chunks_b])
            continue
            
        b_res = []
        for chunk_b in chunks_b:
            chunk_scores = [float(Levenshtein.distance(chunk_a, chunk_b)) for chunk_a in chunks_a]
            
            if operation == 'max':
                b_res.append(max(chunk_scores))
            elif operation == 'min':
                b_res.append(min(chunk_scores))
            elif operation == 'mean':
                b_res.append(sum(chunk_scores) / len(chunk_scores))
            else:
                raise ValueError(f"Unrecognized operation: {operation}")
        results.append(b_res)
    return results

def chunkwise_quantile(values_list: list[list[float]]) -> list[list[float]]:
    """Compute quantile for a list of lists of values."""
    flat_values = [v for values in values_list for v in values]
    if not flat_values:
        return [[] for _ in values_list]
        
    n = len(flat_values)
    if n == 1:
        return [[1.0 for _ in values] for values in values_list]
        
    arr = np.array(flat_values, dtype=float)
    sorted_arr = np.sort(arr)
    
    results = []
    for values in values_list:
        if not values:
            results.append([])
            continue
        v_arr = np.array(values, dtype=float)
        left_ranks = np.searchsorted(sorted_arr, v_arr, side='left')
        right_ranks = np.searchsorted(sorted_arr, v_arr, side='right')
        avg_ranks = (left_ranks + 1 + right_ranks) / 2.0
        results.append((avg_ranks / n).tolist())
        
    return results

def chunkwise_min_max_norm(values_list: list[list[float]]) -> list[list[float]]:
    """Compute min-max normalization for a list of lists of values."""
    flat_values = [v for values in values_list for v in values]
    if not flat_values:
        return [[] for _ in values_list]
        
    arr = np.array(flat_values, dtype=float)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val == max_val:
        return [[0.0 for _ in values] for values in values_list]
        
    results = []
    for values in values_list:
        if not values:
            results.append([])
            continue
        v_arr = np.array(values, dtype=float)
        norm_v = (v_arr - min_val) / (max_val - min_val)
        results.append(norm_v.tolist())
        
    return results