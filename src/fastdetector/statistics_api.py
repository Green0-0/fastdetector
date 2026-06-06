import asyncio
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional

async def _fetch_logprobs_async(client: AsyncOpenAI, model_name: str, text: str, top_logprobs_k: int, sem: asyncio.Semaphore):
    async with sem:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=text,
                max_tokens=1,
                echo=True,
                logprobs=top_logprobs_k,
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

async def _batch_fetch_logprobs_async(api_url: str, texts: list[str], top_logprobs_k: int, concurrency: int = 256):
    api_url = api_url.rstrip("/")
    client = AsyncOpenAI(base_url=api_url, api_key="EMPTY", max_retries=5, timeout=360.0)
    try:
        models = await client.models.list(timeout=5.0)
        model_name = models.data[0].id
    except Exception:
        model_name = "default-model"
        
    sem = asyncio.Semaphore(concurrency)
    total = len(texts)
    completed = 0
    
    async def _tracked(text):
        nonlocal completed
        if not text.strip():
            result = {}
        else:
            result = await _fetch_logprobs_async(client, model_name, text, top_logprobs_k, sem)
        completed += 1
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} requests complete", flush=True)
        return result
        
    tasks = [_tracked(t) for t in texts]
    results = await asyncio.gather(*tasks)
    await client.close()
    return results

def fetch_logprobs_all(texts: list[str], api_url: str, top_logprobs_k: int = 100, concurrency: int = 256) -> tuple[list[list[Optional[float]]], list[list[dict[str, float]]]]:
    """Fetch logprobs for a list of texts using the vLLM API.
    
    Args:
        texts: List of strings.
        api_url: The vLLM API URL.
        top_logprobs_k: The number of top logprobs to fetch per token.
        concurrency: Max concurrent API requests.
        
    Returns:
        Tuple of (token_logprobs_list, top_logprobs_list)
    """
    results = asyncio.run(_batch_fetch_logprobs_async(api_url, texts, top_logprobs_k, concurrency))
    
    token_logprobs_list = []
    top_logprobs_list = []
    for r in results:
        token_logprobs_list.append(r.get("token_logprobs", []))
        top_logprobs_list.append(r.get("top_logprobs", []))
        
    return token_logprobs_list, top_logprobs_list

def batch_gen_embeddings(texts: list[str], model_name: str = "Qwen/Qwen3-Embedding-4B", batch_size: int = 32) -> np.ndarray:
    """Generate normalized embeddings for a list of texts.
    
    Args:
        texts: List of strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        
    Returns:
        Numpy array of normalized embeddings.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def batch_cross_encoder(texts_a: list[str], texts_b: list[str], model_name: str = "Qwen/Qwen3-Reranker-4B", batch_size: int = 4) -> list[float]:
    """Compute cross-encoder scores for aligned pairs of texts.
    
    Args:
        texts_a: First list of texts.
        texts_b: Second list of texts.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        
    Returns:
        List of cross-encoder scores.
    """
    model = CrossEncoder(model_name, trust_remote_code=True)
    pairs = list(zip(texts_a, texts_b))
    scores = model.predict(pairs, batch_size=batch_size)
    return scores.tolist()
