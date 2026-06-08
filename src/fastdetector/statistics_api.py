import asyncio
import numpy as np
import torch
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
                timeout=600.0
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

async def _batch_fetch_logprobs_async(api_url: str, texts: list[str], top_logprobs_k: int, concurrency: int = 256, user_prefill: Optional[str] = None):
    if user_prefill is None:
        user_prefill = "Write me a document or piece of web text that you remember seeing from your training data, of your choice. The topic, formatting, and content are yours to decide."

    api_url = api_url.rstrip("/")
    client = AsyncOpenAI(base_url=api_url, api_key="EMPTY", max_retries=5, timeout=600.0)
    try:
        models = await client.models.list(timeout=5.0)
        model_name = models.data[0].id
    except Exception:
        model_name = "default-model"
        
    num_prefix_tokens = 0
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Warning: Could not load tokenizer for {model_name}: {e}. Falling back to raw text.")

    sem = asyncio.Semaphore(concurrency)
    total = len(texts)
    completed = 0
    
    async def _tracked(text):
        nonlocal completed
        if not text.strip():
            result = {}
        else:
            if tokenizer:
                messages = [
                    {"role": "user", "content": user_prefill},
                    {"role": "assistant", "content": text}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                
                prefix_messages = [
                    {"role": "user", "content": user_prefill},
                    {"role": "assistant", "content": ""}
                ]
                prefix_prompt = tokenizer.apply_chat_template(prefix_messages, tokenize=False)
                num_prefix_tokens = len(tokenizer.encode(prefix_prompt, add_special_tokens=False))
            else:
                formatted_prompt = text
                num_prefix_tokens = 0

            result = await _fetch_logprobs_async(client, model_name, formatted_prompt, top_logprobs_k, sem)
            if result and num_prefix_tokens > 0:
                for key in ["token_logprobs", "top_logprobs", "tokens"]:
                    if result.get(key) and len(result[key]) > num_prefix_tokens:
                        result[key] = result[key][num_prefix_tokens:]
        completed += 1
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} requests complete", flush=True)
        return result
        
    tasks = [_tracked(t) for t in texts]
    results = await asyncio.gather(*tasks)
    await client.close()
    return results

def fetch_logprobs_all(texts: list[str], api_url: str, top_logprobs_k: int = 100, concurrency: int = 256, user_prefill: Optional[str] = None) -> tuple[list[list[Optional[float]]], list[list[dict[str, float]]]]:
    """Fetch logprobs for a list of texts using the vLLM API.
    
    Args:
        texts: List of strings.
        api_url: The vLLM API URL.
        top_logprobs_k: The number of top logprobs to fetch per token.
        concurrency: Max concurrent API requests.
        user_prefill: Optional user message prefill.
        
    Returns:
        Tuple of (token_logprobs_list, top_logprobs_list)
    """
    results = asyncio.run(_batch_fetch_logprobs_async(api_url, texts, top_logprobs_k, concurrency, user_prefill))
    
    token_logprobs_list = []
    top_logprobs_list = []
    for r in results:
        token_logprobs_list.append(r.get("token_logprobs", []))
        top_logprobs_list.append(r.get("top_logprobs", []))
        
    return token_logprobs_list, top_logprobs_list

def batch_gen_embeddings(texts: list[str], model_name: str = "Qwen/Qwen3-Embedding-4B", batch_size: int = 4) -> np.ndarray:
    """Generate normalized embeddings for a list of texts.
    
    Args:
        texts: List of strings.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        
    Returns:
        Numpy array of normalized embeddings.
    """
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
        kwargs["processor_kwargs"] = {"padding_side": "left"}

    model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings

def batch_cross_encoder(texts_a: list[str], texts_b: list[str], model_name: str = "Qwen/Qwen3-Reranker-4B", batch_size: int = 2) -> list[float]:
    """Compute cross-encoder scores for aligned pairs of texts.
    
    Args:
        texts_a: First list of texts.
        texts_b: Second list of texts.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for inference.
        
    Returns:
        List of cross-encoder scores.
    """
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
        kwargs["processor_kwargs"] = {"padding_side": "left"}

    model = CrossEncoder(model_name, trust_remote_code=True, **kwargs)
    pairs = list(zip(texts_a, texts_b))
    scores = model.predict(pairs, batch_size=batch_size)
    return scores.tolist()