import math
import numpy as np
import torch
import scipy.optimize
import ot
from collections import Counter
import Levenshtein
from typing import Optional
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

def fastdetectgpt_scores_approx(texts: list[str], token_logprobs: list[list[Optional[float]]], top_logprobs: list[list[dict[str, float]]]) -> list[float]:
    """Approximate FastDetectGPT score for each text using top-N logprobs.
    Score = mean((log_prob(x_i) - E[log_prob(x_i)]) / Std[log_prob(x_i)])
    
    Args:
        texts: List of strings.
        token_logprobs: Logprobs of the actual tokens.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        
    Returns:
        List of approximated FastDetectGPT scores.
    """
    results = []
    for text, text_token_lps, text_top_lps in zip(texts, token_logprobs, top_logprobs):
        if not text.strip() or not text_token_lps or not text_top_lps:
            results.append(0.0)
            continue
            
        total_lp = 0.0
        total_expected_lp = 0.0
        total_variance = 0.0
        valid_tokens = 0
        
        for lp, top_lps in zip(text_token_lps, text_top_lps):
            if lp is None or top_lps is None or len(top_lps) == 0:
                continue
                
            p = np.array([math.exp(v) for v in top_lps.values()])
            log_p = np.log(p + 1e-12)
            
            h_top = -np.sum(p * log_p)
            var_top = np.sum(p * (log_p ** 2))
            
            Z = np.sum(p)
            M = max(0.0, 1.0 - Z)
            
            if M > 0:
                p_min = np.min(p)
                p_bound = min(p_min, M)
                log_p_tail = math.log(p_bound + 1e-12)
                h_tail = -M * log_p_tail
                var_tail = M * (log_p_tail ** 2)
            else:
                h_tail = 0.0
                var_tail = 0.0
                
            expected_lp = -(h_top + h_tail)
            expected_lp_sq = var_top + var_tail
            
            variance = max(0.0, expected_lp_sq - (expected_lp ** 2))
            
            total_lp += lp
            total_expected_lp += expected_lp
            total_variance += variance
            valid_tokens += 1
            
        if valid_tokens > 0 and total_variance > 1e-6:
            sequence_score = (total_lp - total_expected_lp) / math.sqrt(total_variance)
            results.append(sequence_score)
        else:
            results.append(0.0)
            
    return results

def binoculars_scores_approx(texts: list[str],
                             token_logprobs_m1: list[list[Optional[float]]], 
                             top_logprobs_m1: list[list[dict[str, float]]],
                             top_logprobs_m2: list[list[dict[str, float]]]) -> list[float]:
    """Approximate Binoculars score using top-N logprobs.
    B = log(PPL_M1) / log(X-PPL_M1_M2)
    log(PPL_M1) = - 1/N sum log p_M1(x_i)
    log(X-PPL_M1_M2) = 1/N sum_i H(M2_i, M1_i)
    H(M2_i, M1_i) = - sum_v p_M2(v) log p_M1(v)
    
    Args:
        texts: List of strings.
        token_logprobs_m1: Actual token logprobs from M1 (Observer).
        top_logprobs_m1: Top logprobs dicts from M1 (Observer).
        top_logprobs_m2: Top logprobs dicts from M2 (Performer).
        
    Returns:
        List of approximated Binoculars scores.
    """
    results = []
    for text, token_lps_m1, top_lps_m1, top_lps_m2 in zip(texts, token_logprobs_m1, top_logprobs_m1, top_logprobs_m2):
        if not text.strip() or not token_lps_m1 or not top_lps_m1 or not top_lps_m2:
            results.append(0.0)
            continue
            
        total_lp_m1 = 0.0
        total_cross_entropy = 0.0
        valid_tokens = 0
        
        for lp_m1, top_m1, top_m2 in zip(token_lps_m1, top_lps_m1, top_lps_m2):
            if lp_m1 is None or not top_m1 or not top_m2:
                continue
                
            p_m1_dict = {k: math.exp(v) for k, v in top_m1.items()}
            p_m2_dict = {k: math.exp(v) for k, v in top_m2.items()}
            
            Z_m1 = sum(p_m1_dict.values())
            M_m1 = max(0.0, 1.0 - Z_m1)
            p_min_m1 = min(p_m1_dict.values()) if p_m1_dict else 0.0
            p_bound_m1 = min(p_min_m1, M_m1)
            lp_tail_m1 = math.log(p_bound_m1 + 1e-12)
            
            Z_m2 = sum(p_m2_dict.values())
            M_m2 = max(0.0, 1.0 - Z_m2)
            
            cross_entropy = 0.0
            for v, p_m2_v in p_m2_dict.items():
                if v in top_m1:
                    lp_m1_v = top_m1[v]
                else:
                    lp_m1_v = lp_tail_m1
                cross_entropy -= p_m2_v * lp_m1_v
                
            cross_entropy -= M_m2 * lp_tail_m1
            
            total_lp_m1 += lp_m1
            total_cross_entropy += cross_entropy
            valid_tokens += 1
            
        if valid_tokens > 0 and total_cross_entropy > 1e-6:
            results.append(-total_lp_m1 / total_cross_entropy)
        else:
            results.append(0.0)
            
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

def _compute_idf(tokens_lists: list[list[str]]) -> dict[str, float]:
    df = Counter()
    for tokens in tokens_lists:
        for token in set(tokens):
            df[token] += 1
    N = max(1, len(tokens_lists))
    return {token: math.log((N + 1) / (count + 1)) for token, count in df.items()}

def _get_idf_weights(tokens: list[str], idf_dict: dict[str, float]) -> np.ndarray:
    if not tokens:
        return np.array([])
    w = np.array([idf_dict.get(token, 0.0) for token in tokens])
    w = np.maximum(w, 0.0)
    if w.sum() <= 1e-12:
        w = np.ones(len(tokens))
    return w / w.sum()

def bertscore(src_embeddings_list: list[np.ndarray | torch.Tensor], edit_embeddings_list: list[np.ndarray | torch.Tensor], src_tokens_list: Optional[list[list[str]]] = None, edit_tokens_list: Optional[list[list[str]]] = None) -> tuple[list[float], list[float], list[float]]:
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
            
        sim_matrix = torch.mm(edit, src.t())  # M x N
        
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
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return precisions, recalls, f1s

def moverscore(src_embeddings_list: list[np.ndarray | torch.Tensor], edit_embeddings_list: list[np.ndarray | torch.Tensor], src_tokens_list: Optional[list[list[str]]] = None, edit_tokens_list: Optional[list[list[str]]] = None) -> list[float]:
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
            
        N, D = src.shape
        M, _ = edit.shape
        
        if N == 0 or M == 0:
            results.append(1.0)
            continue
            
        sim_matrix = src @ edit.T
        
        # Euclidean distance for normalized embeddings is sqrt(max(0, 2 - 2 * cos_sim))
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