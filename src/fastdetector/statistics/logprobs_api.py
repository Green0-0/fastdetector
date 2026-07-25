import asyncio
from typing import Optional
from openai import AsyncOpenAI


async def _fetch_logprobs_async(
    client: AsyncOpenAI,
    model_name: str,
    text: str,
    top_logprobs_k: int,
    sem: asyncio.Semaphore,
):
    async with sem:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=text,
                max_tokens=1,
                echo=True,
                logprobs=top_logprobs_k,
                timeout=600.0,
            )
            logprobs = response.choices[0].logprobs.model_dump()
            if logprobs:
                for key in ("token_logprobs", "top_logprobs", "tokens"):
                    if logprobs.get(key):
                        logprobs[key] = logprobs[key][:-1]
            return logprobs
        except Exception as e:
            print(f"Error fetching logprobs: {e}")
            return {}


async def _batch_fetch_logprobs_async(
    api_url: str,
    texts: list[str],
    top_logprobs_k: int,
    concurrency: int = 256,
):
    api_url = api_url.rstrip("/")
    client = AsyncOpenAI(base_url=api_url, api_key="EMPTY", max_retries=5, timeout=600.0)
    try:
        models = await client.models.list(timeout=5.0)
        model_name = models.data[0].id
    except Exception as e:
        await client.close()
        raise RuntimeError(
            f"Could not list models from {api_url}/v1/models — server may be "
            f"unreachable or misconfigured. Original error: {e}"
        ) from e

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
    try:
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()
    return results


def fetch_logprobs_all(
    texts: list[str],
    api_url: str,
    top_logprobs_k: int = 100,
    concurrency: int = 256,
) -> tuple[list[list[Optional[float]]], list[list[dict[str, float]]]]:
    """Fetch logprobs for a list of texts using the vLLM completions API.

    Args:
        texts: List of strings.
        api_url: The vLLM API URL (e.g. "http://localhost:8000/v1").
        top_logprobs_k: The number of top logprobs to fetch per token.
        concurrency: Max concurrent API requests.

    Returns:
        Tuple of (token_logprobs_list, top_logprobs_list).
        - token_logprobs_list[i] is a list of logprobs for the actual tokens
          in texts[i] (None for the first position).
        - top_logprobs_list[i] is a list of dicts mapping top tokens to their
          logprobs, one dict per position.
    """
    results = asyncio.run(_batch_fetch_logprobs_async(api_url, texts, top_logprobs_k, concurrency))

    token_logprobs_list = []
    top_logprobs_list = []
    for r in results:
        token_logprobs_list.append(r.get("token_logprobs", []))
        top_logprobs_list.append(r.get("top_logprobs", []))

    return token_logprobs_list, top_logprobs_list
