import math
import numpy as np
from typing import Optional

def entropies_approx(top_logprobs: list[list[dict[str, float]]]) -> list[float]:
    """Approximate mean next-token entropy for each text using top-N logprobs and a tail-mass heuristic.
    
    Args:
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        
    Returns:
        List of approximated entropy values.
    """
    results = []
    for text_top_lps in top_logprobs:
        if not text_top_lps:
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

def fastdetectgpt_scores_approx(token_logprobs: list[list[Optional[float]]], top_logprobs: list[list[dict[str, float]]]) -> list[float]:
    """Approximate FastDetectGPT score for each text using top-N logprobs.
    Score = mean((log_prob(x_i) - E[log_prob(x_i)]) / Std[log_prob(x_i)])
    
    Args:
        token_logprobs: Logprobs of the actual tokens.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        
    Returns:
        List of approximated FastDetectGPT scores.
    """
    results = []
    for text_token_lps, text_top_lps in zip(token_logprobs, top_logprobs):
        if not text_token_lps or not text_top_lps:
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

def binoculars_scores_approx(token_logprobs_m1: list[list[Optional[float]]], 
                             top_logprobs_m1: list[list[dict[str, float]]],
                             top_logprobs_m2: list[list[dict[str, float]]]) -> list[float]:
    """Approximate Binoculars score using top-N logprobs.
    B = log(PPL_M1) / log(X-PPL_M1_M2)
    log(PPL_M1) = - 1/N sum log p_M1(x_i)
    log(X-PPL_M1_M2) = 1/N sum_i H(M2_i, M1_i)
    H(M2_i, M1_i) = - sum_v p_M2(v) log p_M1(v)
    
    Args:
        token_logprobs_m1: Actual token logprobs from M1 (Observer).
        top_logprobs_m1: Top logprobs dicts from M1 (Observer).
        top_logprobs_m2: Top logprobs dicts from M2 (Performer).
        
    Returns:
        List of approximated Binoculars scores.
    """
    results = []
    for token_lps_m1, top_lps_m1, top_lps_m2 in zip(token_logprobs_m1, top_logprobs_m1, top_logprobs_m2):
        if not token_lps_m1 or not top_lps_m1 or not top_lps_m2:
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

def perplexities(token_logprobs: list[list[Optional[float]]]) -> list[float]:
    """Compute perplexity for each text.
    
    Args:
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        
    Returns:
        List of perplexity values.
    """
    results = []
    for text_token_lps in token_logprobs:
        if not text_token_lps:
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

def top_p_outlier_percentages(top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[Optional[float]]], p: float) -> list[float]:
    """Compute the percentage of tokens outside the top-p probability mass for each text.
    
    Args:
        top_logprobs: For each text, a list of dictionaries mapping top tokens to logprobs.
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        p: Probability mass threshold (e.g., 0.9).
        
    Returns:
        List of outlier percentages.
    """
    results = []
    for text_top_lps, text_token_lps in zip(top_logprobs, token_logprobs):
        if not text_top_lps or not text_token_lps:
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

def top_k_outlier_percentages(top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[Optional[float]]], k: int) -> list[float]:
    """Compute the percentage of tokens outside the top-k probability mass for each text.
    
    Args:
        top_logprobs: For each text, a list of dictionaries mapping top tokens to logprobs.
        token_logprobs: For each text, a list of logprobs for the actual tokens.
        k: Top-k threshold.
        
    Returns:
        List of outlier percentages.
    """
    results = []
    for text_top_lps, text_token_lps in zip(top_logprobs, token_logprobs):
        if not text_top_lps or not text_token_lps:
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