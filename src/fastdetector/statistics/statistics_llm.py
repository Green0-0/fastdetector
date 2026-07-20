"""LLM-derived text metrics computed from top-logprobs (entropy, FastDetectGPT, Binoculars)."""

import math
import numpy as np

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

def fastdetectgpt_scores_approx(token_logprobs: list[list[float | None]], top_logprobs: list[list[dict[str, float]]]) -> list[float]:
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

def binoculars_scores_approx(token_logprobs_m1: list[list[float | None]],
                             top_logprobs_m1: list[list[dict[str, float]]],
                             top_logprobs_m2: list[list[dict[str, float]]],
                             token_logprobs_m2: list[list[float | None]]) -> list[float]:
    """Approximate Binoculars score, matching the official Hans et al. (2024) implementation.

    Official formula (https://github.com/ahans30/Binoculars/blob/main/binoculars/detector.py):

        B(x) = PPL_performer(x) / X-PPL(x)
             = (-mean log p_M2(x_i)) / (mean H(p_M1, p_M2))
             = (-mean log p_M2(x_i)) / (mean_i -sum_v p_M1(v) log p_M2(v))

    where M1 = observer, M2 = performer. The score is in **nats**
    (i.e., the ratio of cross-entropies, not the ratio of exponentiated
    perplexities) — this matches the official code, which computes
    `ppl / x_ppl` where both `ppl` and `x_ppl` are mean cross-entropies
    in nats.

    Direction (verified against the official repo's README example,
    which reports B = 0.7566 for ChatGPT text vs the 0.9013 threshold):

        - AI-generated text  → B < 1 (performer is confident on its own outputs)
        - human-written text → B > 1 (performer is less confident)

    The previous implementation in this file computed
    `PPL_observer / H(p_M2, p_M1)` — i.e. the observer/performer roles
    were swapped in both the numerator and the denominator. That is a
    *different* formula from the paper and from the official code, and
    does not reproduce the official threshold of ~0.9013 used for
    detection.

    Top-N logprobs approximation: only the top-N entries of each
    distribution are available, so the cross-entropy
    `H(p_M1, p_M2) = -sum_v p_M1(v) log p_M2(v)` is approximated by:

      * For v in M1's top-N with v also in M2's top-N: use the exact
        `log p_M2(v)` from `top_m2[v]`.
      * For v in M1's top-N but NOT in M2's top-N: use the tail bound
        `lp_tail_m2 = log(min(p_min_m2, M_m2))` (an upper bound on
        `log p_M2(v)` for v outside M2's top-N, since `p_M2(v) <= p_min_m2`
        whenever v is not in the top-N).
      * For v in M1's tail (total mass `M_m1`): assume all tail mass
        sits at `lp_tail_m2`, contributing `-M_m1 * lp_tail_m2`.

    This tail heuristic is the same one used by `entropies_approx` and
    `fastdetectgpt_scores_approx`; it is a known approximation, not a
    closed-form solution.

    Args:
        token_logprobs_m1: Actual token logprobs from M1 (Observer).
            (Unused by the official formula but kept because the
            caller has it readily available and to keep the function
            signature self-documenting about which model is which.)
        top_logprobs_m1: Top logprobs dicts from M1 (Observer).
        top_logprobs_m2: Top logprobs dicts from M2 (Performer).
        token_logprobs_m2: Actual token logprobs from M2 (Performer).
            Required for the numerator (PPL_M2).

    Returns:
        List of Binoculars scores (PPL_M2 / H(p_M1, p_M2), in nats).
        Lower = more AI-like; the official accuracy threshold is ~0.9013.
    """
    results = []
    for token_lps_m2, top_lps_m1, top_lps_m2, token_lps_m1 in zip(
        token_logprobs_m2, top_logprobs_m1, top_logprobs_m2, token_logprobs_m1
    ):
        # Validate inputs. We need M1's top_logprobs (for the denominator),
        # M2's top_logprobs (for the denominator tail bound), and M2's token
        # logprobs (for the numerator).
        if not top_lps_m1 or not top_lps_m2 or not token_lps_m2:
            results.append(0.0)
            continue

        total_neg_lp_m2 = 0.0
        total_cross_entropy = 0.0
        valid_tokens = 0

        for lp_m2, top_m1, top_m2, lp_m1 in zip(
            token_lps_m2, top_lps_m1, top_lps_m2, token_lps_m1
        ):
            if lp_m2 is None or not top_m1 or not top_m2 or lp_m1 is None:
                continue

            # --- Build distributions / tail bounds ---
            p_m1_dict = {k: math.exp(v) for k, v in top_m1.items()}
            p_m2_dict = {k: math.exp(v) for k, v in top_m2.items()}

            # Tail bound for M2: an upper bound on log p_M2(v) for any v
            # not in M2's top-N. Since p_M2(v) <= p_min_m2 for such v
            # (otherwise v would be in the top-N), we have
            # log p_M2(v) <= log p_min_m2. We also cap by M_m2 (tail mass)
            # to handle the degenerate single-tail-token case.
            Z_m2 = sum(p_m2_dict.values())
            M_m2 = max(0.0, 1.0 - Z_m2)
            p_min_m2 = min(p_m2_dict.values()) if p_m2_dict else 0.0
            p_bound_m2 = min(p_min_m2, M_m2)
            lp_tail_m2 = math.log(p_bound_m2 + 1e-12)

            # Tail mass for M1 (used to weight the lump tail contribution
            # in the cross-entropy denominator).
            Z_m1 = sum(p_m1_dict.values())
            M_m1 = max(0.0, 1.0 - Z_m1)

            # --- Compute H(p_M1, p_M2) = -sum_v p_M1(v) log p_M2(v) ---
            # Iterate over M1's top-N tokens. For each, look up log p_M2(v)
            # in M2's top-N if present, else use the M2 tail bound.
            cross_entropy = 0.0
            for v, p_m1_v in p_m1_dict.items():
                if v in top_m2:
                    lp_m2_v = top_m2[v]
                else:
                    lp_m2_v = lp_tail_m2
                cross_entropy -= p_m1_v * lp_m2_v

            # Add the contribution from M1's tail (mass M_m1) — assume all
            # tail mass sits at the M2 tail bound, same heuristic as
            # entropies_approx / fastdetectgpt_scores_approx.
            cross_entropy -= M_m1 * lp_tail_m2

            # --- Accumulate numerator (PPL_M2) and denominator (X-PPL) ---
            total_neg_lp_m2 += -lp_m2  # = -log p_M2(x_i)
            total_cross_entropy += cross_entropy
            valid_tokens += 1

        if valid_tokens > 0 and total_cross_entropy > 1e-6:
            # B = PPL_M2 / X-PPL (both in nats — ratio of mean cross-entropies,
            # matching the official implementation).
            results.append(total_neg_lp_m2 / total_cross_entropy)
        else:
            results.append(0.0)

    return results

def perplexities(token_logprobs: list[list[float | None]]) -> list[float]:
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

def top_p_outlier_percentages(top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[float | None]], p: float) -> list[float]:
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

def top_k_outlier_percentages(top_logprobs: list[list[dict[str, float]]], token_logprobs: list[list[float | None]], k: int) -> list[float]:
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