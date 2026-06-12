import asyncio
import numpy as np
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional

from fastdetector.statistics.statistics_basic import extract_ngrams

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

async def _batch_fetch_logprobs_async(api_url: str, texts: list[str], top_logprobs_k: int, concurrency: int = 256):
    api_url = api_url.rstrip("/")
    client = AsyncOpenAI(base_url=api_url, api_key="EMPTY", max_retries=5, timeout=600.0)
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
    return (-1.0 * scores).tolist()

def batch_soft_ngram_scores(
    source_texts: list[str],
    edited_texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.8,
    min_length: int = 6,
    max_length: int = 12,
    phrase_batch_size: int = 2048,
) -> list[float]:
    """Compute soft n-gram distance between pairs of source and edited texts.
    
    Args:
        source_texts: List of original texts.
        edited_texts: List of edited/generated texts.
        model_name: HuggingFace sentence-transformers model ID.
        threshold: Cosine similarity threshold for counting a phrase as matched.
        min_length: Minimum n-gram length in words.
        max_length: Maximum n-gram length in words.
        phrase_batch_size: Encoding batch size for soft n-gram phrases.
        
    Returns:
        List of distance scores (1 - precision).
    """
    kwargs = {}
    if "qwen3" in model_name.lower():
        kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
        kwargs["processor_kwargs"] = {"padding_side": "left"}

    model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)
    
    results = []
    src_phrases_list = extract_ngrams(source_texts, min_length=min_length, max_length=max_length)
    edit_phrases_list = extract_ngrams(edited_texts, min_length=min_length, max_length=max_length)
    
    unique_phrases = list(set(
        phrase 
        for phrases in src_phrases_list + edit_phrases_list 
        for phrase in phrases
    ))
    
    print(f"Encoding {len(unique_phrases)} unique soft n-grams with batch size {phrase_batch_size}...")
    
    all_embeddings = model.encode(
        unique_phrases, 
        convert_to_tensor=True, 
        batch_size=phrase_batch_size, 
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    phrase_to_idx = {phrase: i for i, phrase in enumerate(unique_phrases)}
    
    for src_phrases, edit_phrases in zip(src_phrases_list, edit_phrases_list):
        if not src_phrases or not edit_phrases:
            results.append(1.0)
            continue
            
        src_indices = [phrase_to_idx[p] for p in src_phrases]
        edit_indices = [phrase_to_idx[p] for p in edit_phrases]
        
        src_embeddings = all_embeddings[src_indices]
        edit_embeddings = all_embeddings[edit_indices]
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(src_embeddings, edit_embeddings.t())
        
        # Fraction of edited phrases that match any source phrase
        matches = (similarity_matrix >= threshold).any(dim=0)
        precision = matches.sum().item() / len(edit_phrases)
        
        results.append(1.0 - precision)
        
    return results

def batch_extract_token_embeddings(
    texts: list[str],
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 4,
) -> tuple[list[torch.Tensor], list[list[str]]]:
    """Extract normalized token-level embeddings and subword tokens using a bidirectional encoder.
    
    Args:
        texts: List of texts.
        model_name: HuggingFace model identifier for a bidirectional encoder.
        batch_size: Inference batch size.
        
    Returns:
        Tuple of (embs, tokens) where:
            - embs is a list of NxD tensors.
            - tokens is a list of lists of subword strings.
    """
    from transformers import AutoModel, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    all_embs = []
    all_tokens = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=512)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = model(**inputs)
            
        batch_embs = []
        batch_tokens = []
        
        for b_idx in range(len(batch_texts)):
            mask = inputs['attention_mask'][b_idx]
            length = mask.sum().item()
            
            if length > 2:
                embs = outputs.last_hidden_state[b_idx, 1:length-1]
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                batch_embs.append(embs)
                
                ids = inputs['input_ids'][b_idx, 1:length-1]
                batch_tokens.append(tokenizer.convert_ids_to_tokens(ids))
            else:
                batch_embs.append(torch.empty((0, outputs.last_hidden_state.size(-1))))
                batch_tokens.append([])
                
        all_embs.extend([emb.cpu() for emb in batch_embs])
        all_tokens.extend(batch_tokens)
            
    return all_embs, all_tokens