"""LLM-derived text metrics computed from top-logprobs (entropy, FastDetectGPT, Binoculars)."""

import math
import numpy as np

# Default assumption for the size of the vocabulary outside the top-N
# logprobs returned by the server. Used by the tail-mass heuristic in
# ``entropies_approx`` / ``fastdetectgpt_scores_approx`` to estimate the
# entropy (and second moment) contribution of the tail.
#
# This is intentionally a conservative default — most modern LLMs have
# vocabularies in the 30k-130k range (Llama-3: 128k, Qwen-2.5: 152k,
# GPT-2: 50k). The heuristic's bias scales with ``log(V - N)``, so
# choosing a value in the middle of the typical range keeps the bias
# small across models. Callers can override via the ``vocab_size``
# parameter to ``entropies_approx`` / ``fastdetectgpt_scores_approx``
# if they know the exact vocabulary size of the model serving logprobs.
DEFAULT_VOCAB_SIZE = 128000


def _tail_moments(p: np.ndarray, vocab_size: int | None) -> tuple[float, float, float]:
    """Estimate the tail's contribution to H, E[(log p)^2], and mass.

    Given the top-N probabilities ``p`` (sum <= 1), estimate the tail
    mass ``M = 1 - sum(p)`` and its contributions to:

    - the entropy ``H_tail = -sum_{v in tail} p(v) log p(v)``
    - the second moment ``E_tail[(log p)^2] = sum_{v in tail} p(v) (log p(v))^2``

    Two approximations are supported:

    - **Uniform-tail** (``vocab_size`` provided): assume the tail mass
      ``M`` is distributed uniformly over ``V - N`` tokens, where
      ``V = vocab_size`` and ``N = len(p)``. This is the
      maximum-entropy tail distribution and therefore an **upper bound**
      on the true tail entropy / second moment. It is the standard
      choice when a vocab size is available (e.g. the FastDetectGPT
      reference implementation uses it).
    - **Concentrated-tail** (``vocab_size=None``, the previous default):
      assume the tail mass is split into ``K = M / p_bound`` categories
      each with probability ``p_bound = min(p_min, M)``, where ``p_min``
      is the smallest top-N probability. This is a different specific
      assumption — it is neither an upper nor a lower bound in general.
      For typical LLMs with large vocabularies (V ≈ 30k-130k) and
      top-N=100, ``K`` is usually much smaller than ``V - N``, so this
      heuristic **underestimates** the tail entropy / second moment
      (it assumes fewer tail categories than actually exist). However,
      when ``p_min`` is very small (peaky top-N distributions), ``K``
      can exceed ``V - N`` and the heuristic **overestimates** instead.

    Returns:
        Tuple of ``(h_tail, e2_tail, M)``.
    """
    Z = float(np.sum(p))
    M = max(0.0, 1.0 - Z)
    if M <= 0.0:
        return 0.0, 0.0, 0.0

    if vocab_size is not None:
        n_top = len(p)
        v_tail = max(1, vocab_size - n_top)
        # Uniform tail: each tail token has probability M / v_tail.
        # log(M / v_tail) = log(M) - log(v_tail)
        # H_tail = -v_tail * (M/v_tail) * log(M/v_tail) = -M * log(M/v_tail)
        # E2_tail = v_tail * (M/v_tail) * (log(M/v_tail))^2 = M * (log(M/v_tail))^2
        log_p_tail = math.log(M / v_tail + 1e-12)
        h_tail = -M * log_p_tail
        e2_tail = M * (log_p_tail ** 2)
        return h_tail, e2_tail, M

    # Concentrated-tail heuristic (original behaviour).
    p_min = float(np.min(p)) if len(p) > 0 else 0.0
    p_bound = min(p_min, M)
    log_p_tail = math.log(p_bound + 1e-12)
    h_tail = -M * log_p_tail
    e2_tail = M * (log_p_tail ** 2)
    return h_tail, e2_tail, M


def entropies_approx(
    top_logprobs: list[list[dict[str, float]]],
    vocab_size: int | None = None,
) -> list[float]:
    """Approximate mean next-token entropy for each text using top-N logprobs.

    Args:
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        vocab_size: Optional vocabulary size of the model that produced the
            logprobs. When provided, the tail-mass contribution is estimated
            assuming a uniform distribution over the ``vocab_size - N`` tail
            tokens (a standard maximum-entropy bound). When ``None``, a
            concentrated-tail lower bound is used instead — this is the
            historical default but systematically underestimates the entropy
            for typical LLM vocabularies (30k-130k). Pass ``vocab_size`` if
            you know it.

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
                h_tail, _, _ = _tail_moments(p, vocab_size)
                entropies.append(h_top + h_tail)

        results.append(float(np.mean(entropies)) if entropies else 0.0)
    return results

def fastdetectgpt_scores_approx(
    token_logprobs: list[list[float | None]],
    top_logprobs: list[list[dict[str, float]]],
    vocab_size: int | None = None,
) -> list[float]:
    """Approximate FastDetectGPT score for each text using top-N logprobs.

    Computes the conditional probability curvature as defined in the
    Fast-DetectGPT paper (Bao et al., ICLR 2024), Eq. 3:
        d(x) = (log p(x|x) - mu_tilde) / sigma_tilde

    Using the analytical solution from Appendix B:
        d(x) = (sum_j [log p(x_j|x_{<j}) - mu_j]) / sqrt(sum_j sigma_j^2)
             = (total_lp - total_expected_lp) / sqrt(total_variance)

    This is the z-score of the total conditional log-probability, NOT the
    mean of per-position z-scores. The paper treats the passage as a single
    point and computes the curvature at that point. This matches the official
    implementation at github.com/baoguangsheng/fast-detect-gpt.

    Args:
        token_logprobs: Logprobs of the actual tokens.
        top_logprobs: For each text, a list of dictionaries mapping top tokens to their logprobs.
        vocab_size: Optional vocabulary size for the tail-mass heuristic.
            See :func:`entropies_approx` for details. When ``None``, the
            variance (and therefore the z-score denominator) is
            systematically underestimated, inflating the magnitude of the
            score. Pass ``vocab_size`` for unbiased estimation.

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

            h_tail, e2_tail, _ = _tail_moments(p, vocab_size)

            expected_lp = -(h_top + h_tail)
            expected_lp_sq = var_top + e2_tail

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
                             token_logprobs_m2: list[list[float | None]],
                             top_logprobs_m1: list[list[dict[str, float]]],
                             top_logprobs_m2: list[list[dict[str, float]]]) -> list[float]:
    """Approximate Binoculars score using top-N logprobs.

    Computes the Binoculars score following Hans et al. (2024), Eq. 4:
        B(x) = log(PPL_performer) / X-PPL(observer, performer)

    where:
        log(PPL_performer) = -1/N * sum log p_performer(x_i)
        X-PPL(observer, performer) = 1/N * sum_i H(p_observer_i || p_performer_i)
        H(p_observer_i || p_performer_i) = - sum_v p_observer(v) * log p_performer(v)

    This matches the official implementation at github.com/ahans30/Binoculars,
    which computes:
        ppl = perplexity(encodings, performer_logits)   # log(PPL_performer)
        x_ppl = entropy(observer_logits, performer_logits, ...)  # H(observer, performer)
        binoculars_scores = ppl / x_ppl

    M1 is the Observer and M2 is the Performer, following the convention
    where the first checkpoint is the observer (base model) and the second
    is the performer (instruction-tuned model).

    Note: AI-generated text typically has B < threshold (lower score), while
    human text has B closer to or above 1. The classifier should treat
    human text as the positive class (higher score = human).

    Args:
        token_logprobs_m1: Actual token logprobs from M1 (Observer).
        token_logprobs_m2: Actual token logprobs from M2 (Performer).
        top_logprobs_m1: Top logprobs dicts from M1 (Observer).
        top_logprobs_m2: Top logprobs dicts from M2 (Performer).
        
    Returns:
        List of approximated Binoculars scores.
    """
    results = []
    for token_lps_m1, token_lps_m2, top_lps_m1, top_lps_m2 in zip(
        token_logprobs_m1, token_logprobs_m2, top_logprobs_m1, top_logprobs_m2
    ):
        if not token_lps_m1 or not token_lps_m2 or not top_lps_m1 or not top_lps_m2:
            results.append(0.0)
            continue
            
        total_lp_m2 = 0.0
        total_cross_entropy = 0.0
        valid_tokens = 0
        
        for lp_m1, lp_m2, top_m1, top_m2 in zip(token_lps_m1, token_lps_m2, top_lps_m1, top_lps_m2):
            if lp_m1 is None or lp_m2 is None or not top_m1 or not top_m2:
                continue
                
            p_m1_dict = {k: math.exp(v) for k, v in top_m1.items()}
            p_m2_dict = {k: math.exp(v) for k, v in top_m2.items()}
            
            # Tail mass for M2 (Performer) — used for cross-entropy fallback
            Z_m2 = sum(p_m2_dict.values())
            M_m2 = max(0.0, 1.0 - Z_m2)
            p_min_m2 = min(p_m2_dict.values()) if p_m2_dict else 0.0
            p_bound_m2 = min(p_min_m2, M_m2)
            lp_tail_m2 = math.log(p_bound_m2 + 1e-12)
            
            # Tail mass for M1 (Observer)
            Z_m1 = sum(p_m1_dict.values())
            M_m1 = max(0.0, 1.0 - Z_m1)
            
            # Cross-entropy H(p_observer || p_performer) = -sum_v p_observer(v) * log p_performer(v)
            # We use M1=Observer's distribution to weight M2=Performer's log-probs
            cross_entropy = 0.0
            for v, p_m1_v in p_m1_dict.items():
                if v in top_m2:
                    lp_m2_v = top_m2[v]
                else:
                    lp_m2_v = lp_tail_m2
                cross_entropy -= p_m1_v * lp_m2_v
                
            cross_entropy -= M_m1 * lp_tail_m2
            
            # Numerator: log(PPL_performer) = -sum log p_performer(x_i) / N
            total_lp_m2 += lp_m2
            total_cross_entropy += cross_entropy
            valid_tokens += 1
            
        if valid_tokens > 0 and total_cross_entropy > 1e-6:
            results.append(-total_lp_m2 / total_cross_entropy)
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
            # Perplexity is undefined for empty input.
            # Returning 0.0 ("perfect prediction") is misleading;
            # NaN signals that the value should be excluded from analysis.
            results.append(float('nan'))
            continue
            
        valid_lps = [lp for lp in text_token_lps if lp is not None]
        if not valid_lps:
            results.append(float('nan'))
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
                # Sort top logprobs (not probs) in descending order so we can
                # work in log space — comparing in probability space with an
                # absolute tolerance is incorrect for very small probabilities
                # (e.g. a token_prob of 1e-10 vs threshold_prob of 2e-10 would
                # pass `token_prob < threshold_prob - 1e-6` because the right
                # hand side becomes negative).
                sorted_lps = sorted(top_lps.values(), reverse=True)
                cumulative_prob = 0.0
                threshold_lp = None
                for lp in sorted_lps:
                    cumulative_prob += math.exp(lp)
                    if cumulative_prob >= p:
                        threshold_lp = lp
                        break
                if threshold_lp is None:
                    # Top-N mass did not reach p — fall back to the smallest
                    # top-N logprob as the threshold.
                    threshold_lp = sorted_lps[-1]

                # Compare in log space with a small absolute tolerance. log is
                # monotonic so `token_prob < threshold_prob` is equivalent to
                # `token_lp < threshold_lp`. The tolerance of 1e-5 nats is
                # roughly 0.001% probability ratio, which is well below the
                # numerical precision of fp32 logprobs from vLLM/the OpenAI
                # API, so it only collapses true numerical ties.
                if token_lp < threshold_lp - 1e-5:
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