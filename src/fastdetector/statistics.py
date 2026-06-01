import math
import numpy as np
import asyncio
from openai import OpenAI, AsyncOpenAI
import Levenshtein

from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def _get_logprobs(text: str, api_url: str, top_logprobs: int = 5) -> dict:
    """Helper to get logprobs for a given text from a vLLM completion endpoint using the OpenAI client."""
    api_url = api_url.rstrip("/")
    with OpenAI(base_url=api_url, api_key="EMPTY") as client:
        try:
            models = client.models.list(timeout=5.0)
            model_name = models.data[0].id
        except Exception:
            model_name = "default-model"
            
        response = client.completions.create(
            model=model_name,
            prompt=text,
            max_tokens=1,
            echo=True,
            logprobs=top_logprobs,
            timeout=120.0
        )
        
        logprobs = response.choices[0].logprobs.model_dump()
        
        if logprobs:
            for key in ["token_logprobs", "top_logprobs", "tokens"]:
                if logprobs.get(key):
                    logprobs[key] = logprobs[key][:-1]
                    
        return logprobs

def ngram_analysis(text: str, n: int) -> dict[str, float]:
    """Compute the n-gram analysis for a given text.
    Args:
        text (str): Text to analyze.
        n (int): n-gram size.
    Returns:
        dict[str, float]: Dictionary containing n-gram analysis, with keys being the n-grams and values being their frequencies.
    """
    tokens = text.split()
    if len(tokens) < n:
        return {}
    ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def jacard_ngram(text1: str, text2: str, n: int) -> float:
    """Compute the Jaccard similarity between two texts using n-grams.
    Args:
        text1 (str): First text.
        text2 (str): Second text.
        n (int): n-gram size.
    Returns:
        float: Jaccard similarity.
    """
    t1_tokens = text1.split()
    t2_tokens = text2.split()
    t1 = set([" ".join(t1_tokens[i:i+n]) for i in range(len(t1_tokens) - n + 1)])
    t2 = set([" ".join(t2_tokens[i:i+n]) for i in range(len(t2_tokens) - n + 1)])
    if not t1 and not t2:
        return 1.0 if text1.strip() == text2.strip() else 0.0
    if not t1 or not t2:
        return 0.0
    return len(t1.intersection(t2)) / len(t1.union(t2))

def levenshtein(text1: str, text2: str) -> int:
    """Compute the Levenshtein distance between two texts.
    Args:
        text1 (str): First text.
        text2 (str): Second text.
    Returns:
        int: Levenshtein distance.
    """
    return Levenshtein.distance(text1, text2)

def perplexity(text: str, api_url: str) -> float:
    """Compute the perplexity of a given text, using the model from the api_url (presumably hosted on vLLM with a logits endpoint)."""
    if not text.strip():
        return 0.0
    try:
        logprobs_data = _get_logprobs(text, api_url, top_logprobs=1)
        token_logprobs = logprobs_data.get("token_logprobs", [])
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        if not valid_logprobs:
            return 0.0
        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        return math.exp(-avg_logprob)
    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return 0.0

    except Exception as e:
        print(f"Error computing perplexity: {e}")
        return 0.0

def entropy(text: str, api_url: str) -> float:
    """Approximate mean next-token entropy using top-N logprobs and a tail-mass heuristic."""
    if not text.strip():
        return 0.0
        
    try:
        # Fetch logprobs for the top-100 tokens to approximate the distribution
        logprobs_data = _get_logprobs(text, api_url, top_logprobs=100)
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        entropies = []
        for top_lps in top_logprobs_list:
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
                
        if not entropies:
            return 0.0
        return float(np.mean(entropies))
    except Exception as e:
        print(f"Error computing entropy: {e}")
        return 0.0

    
def top_p_outlier_percentage(text: str, p: float, api_url: str) -> float:
    """Compute the percentage of tokens in the given text which lie outside the top-p probability mass (ie: are extremely unlikely to be sampled), using the model from the api_url (presumably hosted on vLLM with a logits endpoint)."""
    if not text.strip():
        return 0.0
    try:
        logprobs_data = _get_logprobs(text, api_url, top_logprobs=100)
        token_logprobs = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        
        outlier_count = 0
        total_count = 0
        
        for token_lp, top_lps in zip(token_logprobs, top_logprobs_list):
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
                    
        if total_count == 0:
            return 0.0
        return outlier_count / total_count
    except Exception as e:
        print(f"Error computing top_p_outlier_percentage: {e}")
        return 0.0

def top_k_outlier_percentage(text: str, k: int, api_url: str) -> float:
    """Compute the percentage of tokens in the given text which lie outside the top-k probability mass (ie: are extremely unlikely to be sampled), using the model from the api_url (presumably hosted on vLLM with a logits endpoint)."""
    if not text.strip():
        return 0.0
    try:
        logprobs_data = _get_logprobs(text, api_url, top_logprobs=max(100, k))
        token_logprobs = logprobs_data.get("token_logprobs", [])
        top_logprobs_list = logprobs_data.get("top_logprobs", [])
        
        outlier_count = 0
        total_count = 0
        
        for token_lp, top_lps in zip(token_logprobs, top_logprobs_list):
            if token_lp is None or top_lps is None or len(top_lps) < k:
                continue
                
            total_count += 1
            sorted_lps = sorted(top_lps.values(), reverse=True)
            kth_lp = sorted_lps[k-1]
            if token_lp < kth_lp - 1e-5:
                outlier_count += 1
                        
        if total_count == 0:
            return 0.0
        return outlier_count / total_count
    except Exception as e:
        print(f"Error computing top_k_outlier_percentage: {e}")
        return 0.0

def batch_gen_embeddings(dataset):
    """Given a dataset in the format specified by generator.py (one human column, a column for the id of the ai column), compute the embeddings for the human column and the ai column, add them to the dataset in two new columns "human_embeddings" and "ai_embeddings" and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added embeddings columns.
    """
    model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

    def _embed(batch):
        human_texts = batch["original"]
        ai_texts = []
        for i in range(len(batch["original"])):
            idx = batch["final_response_index"][i]
            ai_texts.append(batch[f"response_{idx}"][i])

        return {
            "human_embeddings": model.encode(human_texts, convert_to_numpy=True, normalize_embeddings=True),
            "ai_embeddings": model.encode(ai_texts, convert_to_numpy=True, normalize_embeddings=True),
        }

    return dataset.map(_embed, batched=True, batch_size=32)

def pairwise_cossim_all(dataset):
    """Compute the pairwise cosine similarity between the embeddings of the human and ai columns of a dataset. Add the results to a new column 'pairwise_cossim' and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added pairwise cosine similarity column.
    """
    def _compute(batch):
        human_embs = np.array(batch["human_embeddings"], dtype=np.float32)
        ai_embs = np.array(batch["ai_embeddings"], dtype=np.float32)
        
        # Vectors are pre-normalized, so cosine similarity is just the dot product
        cossims = np.sum(human_embs * ai_embs, axis=1)
        return {"pairwise_cossim": cossims}

    return dataset.map(_compute, batched=True, batch_size=100)

def human_human_cossim_all(dataset):
    """Compute the cosine similarity between a row for the human text against all human texts, averaging the results. Add the results to a new column 'human_human_cossim' and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added pairwise cosine similarity column.
    """
    all_human_embs = np.array(dataset["human_embeddings"], dtype=np.float32)
    
    def _compute(batch):
        batch_embs = np.array(batch["human_embeddings"], dtype=np.float32)
        # Fast BLAS matrix multiplication since vectors are pre-normalized
        sims = batch_embs @ all_human_embs.T
        
        # Exclude the row itself (value is exactly 1.0)
        sum_sims = np.sum(sims, axis=1) - 1.0
        n_items = max(1, len(all_human_embs) - 1)
        
        return {"human_human_cossim": sum_sims / n_items}
        
    return dataset.map(_compute, batched=True, batch_size=100)

def ai_ai_cossim_all(dataset):
    """Compute the pairwise cosine similarity between a row for the ai text against all ai texts, averaging the results. Add the results to a new column 'ai_ai_cossim' and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added pairwise cosine similarity column.
    """
    all_ai_embs = np.array(dataset["ai_embeddings"], dtype=np.float32)
    
    def _compute(batch):
        batch_embs = np.array(batch["ai_embeddings"], dtype=np.float32)
        sims = batch_embs @ all_ai_embs.T
        
        # Exclude the row itself (value is exactly 1.0)
        sum_sims = np.sum(sims, axis=1) - 1.0
        n_items = max(1, len(all_ai_embs) - 1)
        
        return {"ai_ai_cossim": sum_sims / n_items}
        
    return dataset.map(_compute, batched=True, batch_size=100)

def human_ai_cossim_all(dataset):
    """Compute the cosine similarity between a row for the human text against all ai texts, averaging the results. Add the results to a new column 'human_ai_cossim' and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added pairwise cosine similarity column.
    """
    all_ai_embs = np.array(dataset["ai_embeddings"], dtype=np.float32)
    
    def _compute(batch):
        batch_embs = np.array(batch["human_embeddings"], dtype=np.float32)
        sims = batch_embs @ all_ai_embs.T
        return {"human_ai_cossim": np.mean(sims, axis=1)}
        
    return dataset.map(_compute, batched=True, batch_size=100)

def ai_human_cossim_all(dataset):
    """Compute the cosine similarity between a row for the ai text against all human texts, averaging the results. Add the results to a new column 'ai_human_cossim' and return the dataset.
    
    Args:
        dataset (Dataset): Dataset to process.
    Returns:
        Dataset: Dataset with added pairwise cosine similarity column.
    """
    all_human_embs = np.array(dataset["human_embeddings"], dtype=np.float32)
    
    def _compute(batch):
        batch_embs = np.array(batch["ai_embeddings"], dtype=np.float32)
        sims = batch_embs @ all_human_embs.T
        return {"ai_human_cossim": np.mean(sims, axis=1)}
        
    return dataset.map(_compute, batched=True, batch_size=100)

async def _fetch_logprobs_async(client: AsyncOpenAI, model_name: str, text: str, top_logprobs: int, sem: asyncio.Semaphore):
    async with sem:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=text,
                max_tokens=1,
                echo=True,
                logprobs=top_logprobs,
                timeout=120.0
            )
            logprobs = response.choices[0].logprobs.model_dump()
            if logprobs:
                for key in ["token_logprobs", "top_logprobs", "tokens"]:
                    if logprobs.get(key):
                        logprobs[key] = logprobs[key][:-1]
            return logprobs
        except Exception as e:
            print(f"Error fetching logprobs: {e}")
            return {}

def extract_all_stats(logprobs_dict: dict, p: float = 0.9, k: int = 50) -> dict:
    if not logprobs_dict:
        return {"perplexity": 0.0, "entropy": 0.0, "top_p_outlier": 0.0, "top_k_outlier": 0.0}
    
    token_logprobs = logprobs_dict.get("token_logprobs", [])
    top_logprobs_list = logprobs_dict.get("top_logprobs", [])
    
    valid_lps = [lp for lp in token_logprobs if lp is not None]
    try:
        perplexity = math.exp(-sum(valid_lps) / len(valid_lps)) if valid_lps else 0.0
    except OverflowError:
        perplexity = float('inf')
        
    entropies = []
    outlier_p_count = 0
    outlier_k_count = 0
    total_p_count = 0
    total_k_count = 0
    
    for token_lp, top_lps in zip(token_logprobs, top_logprobs_list):
        if token_lp is not None and top_lps is not None and len(top_lps) > 0:
            total_p_count += 1
            
            p_arr = np.array([math.exp(lp) for lp in top_lps.values()])
            h_top = -np.sum(p_arr * np.log(p_arr + 1e-12))
            Z = np.sum(p_arr)
            M = max(0.0, 1.0 - Z)
            h_tail = -M * math.log(min(np.min(p_arr), M) + 1e-12) if M > 0 else 0.0
            entropies.append(h_top + h_tail)
            
            sorted_probs = sorted([math.exp(lp) for lp in top_lps.values()], reverse=True)
            cumulative = 0.0
            threshold_prob = sorted_probs[-1]
            for prob in sorted_probs:
                cumulative += prob
                if cumulative >= p:
                    threshold_prob = prob
                    break
            
            token_prob = math.exp(token_lp)
            if token_prob < threshold_prob - 1e-6:
                outlier_p_count += 1
                
            if len(top_lps) >= k:
                total_k_count += 1
                sorted_lps = sorted(top_lps.values(), reverse=True)
                kth_lp = sorted_lps[k-1]
                if token_lp < kth_lp - 1e-5:
                    outlier_k_count += 1
                    
    entropy_val = float(np.mean(entropies)) if entropies else 0.0
    top_p_outlier = outlier_p_count / total_p_count if total_p_count > 0 else 0.0
    top_k_outlier = outlier_k_count / total_k_count if total_k_count > 0 else 0.0
    
    return {
        "perplexity": perplexity,
        "entropy": entropy_val,
        "top_p_outlier": top_p_outlier,
        "top_k_outlier": top_k_outlier
    }

async def _batch_compute_async(api_url: str, texts: list[str], p: float, k: int):
    api_url = api_url.rstrip("/")
    client = AsyncOpenAI(base_url=api_url, api_key="EMPTY", max_retries=5, timeout=360.0)
    try:
        models = await client.models.list(timeout=5.0)
        model_name = models.data[0].id
    except Exception:
        model_name = "default-model"
        
    sem = asyncio.Semaphore(256)
    top_fetch = max(100, k)
    total = len(texts)
    completed = 0
    
    async def _tracked(text):
        nonlocal completed
        if not text.strip():
            stats = extract_all_stats({}, p, k)
        else:
            lp = await _fetch_logprobs_async(client, model_name, text, top_fetch, sem)
            stats = extract_all_stats(lp, p, k)
        completed += 1
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} requests complete", flush=True)
        return stats
        
    tasks = [_tracked(t) for t in texts]
    results = await asyncio.gather(*tasks)
    await client.close()
    return results

def batch_compute_llm_stats(dataset, api_url: str, p: float = 0.9, k: int = 50):
    human_texts = dataset["original"]
    final_indices = dataset["final_response_index"]
    
    resp_cols = {col: dataset[col] for col in dataset.column_names if col.startswith("response_")}
    ai_texts = [resp_cols[f"response_{idx}"][i] for i, idx in enumerate(final_indices)]
    
    print("Computing stats for human and AI texts concurrently...")
    async def _compute_both():
        t1 = asyncio.create_task(_batch_compute_async(api_url, human_texts, p, k))
        t2 = asyncio.create_task(_batch_compute_async(api_url, ai_texts, p, k))
        return await asyncio.gather(t1, t2)
        
    human_stats, ai_stats = asyncio.run(_compute_both())
    
    for stat_name in human_stats[0].keys():
        dataset = dataset.add_column(f"human_{stat_name}", [s[stat_name] for s in human_stats])
        dataset = dataset.add_column(f"ai_{stat_name}", [s[stat_name] for s in ai_stats])
        
    return dataset