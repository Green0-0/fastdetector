import asyncio
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from fastdetector.batch_state import BatchState
from fastdetector.prompting.prompts import Prompt, PromptSet
from fastdetector.providers import BatchProvider, build_body

MAX_RETRIES = 5

MAX_CONCURRENT = 256


@dataclass
class BatchContext:
    """Everything the offline-batch transport needs beyond the request bodies.

    Args:
        provider: The transport to submit through.
        provider_name: "openai" or "anthropic" - selects the payload dialect.
        state: Durable record of submitted jobs, so a killed run resumes
            polling instead of resubmitting (and re-paying for) a batch.
        max_output_tokens: Output cap; required for Anthropic.
        poll_interval_secs: Delay between status checks.
    """

    provider: BatchProvider
    provider_name: str
    state: BatchState
    max_output_tokens: int | None
    poll_interval_secs: int = 120


async def _send_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    model_name: str = "",
) -> tuple[str, int, int]:
    """Send a single chat completion request.

    Args:
        client: The AsyncOpenAI client.
        semaphore: Bounded semaphore limiting concurrency.
        messages: OpenAI-compatible message list.
        generation_params: Sampling param overrides.
        model_name: Model name for the API call.

    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens).
        On failure, returns ("", 0, 0).
    """
    async with semaphore:
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                **generation_params
            }

            response = await client.chat.completions.create(**kwargs)

            if not response.choices:
                print("Request returned no choices (possibly content-filtered).")
                return "", 0, 0

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            return response.choices[0].message.content or "", prompt_tokens, completion_tokens
        except Exception as e:
            print(f"Request failed: {e}")
            return "", 0, 0


async def _batch_generate_async(
    api_url: str,
    inputs: list[list[dict[str, str]]],
    generation_params: dict[str, Any],
    api_key: str = "EMPTY",
    model_name: str = "",
) -> tuple[list[str], int, int, int]:
    """Fire requests concurrently with a bounded semaphore.

    Args:
        api_url: The URL of the OpenAI-compatible chat completions endpoint
                 (e.g. "http://localhost:8000/v1").
        inputs: A list of conversations, where each conversation is a list of
            {"role": ..., "content": ...} message dicts.
        generation_params: Overrides for sampling params.
        api_key: API key for the endpoint (use "EMPTY" for local vLLM).
        model_name: Model name for the API call.

    Returns:
        A tuple of (list of assistant response strings, total prompt tokens,
        total completion tokens, failed request count). Failed requests appear as empty strings.
    """
    client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        max_retries=MAX_RETRIES,
        timeout=360.0,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(inputs)
    completed = 0
    failed_count = 0

    try:
        async def _tracked_request(messages: list[dict[str, str]]) -> tuple[str, int, int]:
            """Send a request and update progress and failure counts.

            Args:
                messages: List of message dicts.

            Returns:
                Tuple of (response_text, prompt_tokens, completion_tokens).
            """
            nonlocal completed, failed_count
            result = await _send_request(client, semaphore, messages, generation_params, model_name=model_name)
            completed += 1
            if result[0] == "" and result[1] == 0 and result[2] == 0:
                failed_count += 1
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} requests complete", flush=True)
            return result

        tasks = [_tracked_request(messages) for messages in inputs]
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    texts = [r[0] for r in results]
    prompt_tokens = sum(r[1] for r in results)
    completion_tokens = sum(r[2] for r in results)

    if failed_count > 0:
        print(
            f"WARNING: {failed_count}/{total} requests failed (returned empty responses). "
            f"These will appear as empty strings in the output."
        )

    return texts, prompt_tokens, completion_tokens, failed_count


def batch_generate(
    api_url: str,
    inputs: list[list[dict[str, str]]],
    generation_params: dict[str, Any],
    api_key: str = "EMPTY",
    model_name: str = "",
) -> tuple[list[str], int, int, int]:
    """Send a batch of OpenAI-compatible chat conversations concurrently.

    Args:
        api_url: The URL of the OpenAI-compatible chat completions endpoint
                 (e.g. "http://localhost:8000/v1").
        inputs: A list of conversations, where each conversation is a list of
            {"role": ..., "content": ...} message dicts.
        generation_params: Overrides for sampling params.
        api_key: API key for the endpoint (use "EMPTY" for local vLLM).
        model_name: Model name for the API call.

    Returns:
        A tuple of (list of assistant response strings, total prompt tokens,
        total completion tokens, failed request count). Failed requests appear as empty strings.
    """
    return asyncio.run(_batch_generate_async(api_url, inputs, generation_params, api_key, model_name))


def batch_generate_offline(
    ctx: BatchContext,
    inputs: list[list[dict[str, str]]],
    generation_params: dict[str, Any],
    model_name: str,
    turn_index: int,
) -> tuple[list[str], int, int, int]:
    """Run one chat turn through a provider's offline batch API.

    Mirrors :func:`batch_generate`'s contract exactly - same argument shape,
    same ``(texts, prompt_tokens, completion_tokens, failed)`` return, failures
    as aligned empty strings - so callers do not care which transport ran.

    Submitting is the expensive, irreversible step: the job ID is persisted
    before anything else happens, and a run that already has one for this turn
    resumes polling rather than submitting again.

    Args:
        ctx: Provider, state handle, and polling settings.
        inputs: Conversations for the rows active at this turn.
        generation_params: Params already filtered to this provider's accepted set.
        model_name: Model ID, or an Azure deployment name.
        turn_index: Chat turn index, used as the state key.

    Returns:
        A tuple of (response strings, total prompt tokens, total completion
        tokens, failed request count).
    """
    if not inputs:
        return [], 0, 0, 0

    record = ctx.state.get(turn_index)
    if record and record["provider"] == ctx.provider.name:
        job_id = record["job_id"]
        n_requests = record["n_requests"]
        print(f"  Resuming batch for turn {turn_index} from saved job(s).", flush=True)
        if n_requests != len(inputs):
            raise RuntimeError(
                f"Saved batch for turn {turn_index} covers {n_requests} requests but "
                f"this run built {len(inputs)}. The source shard or prompt file changed "
                f"since submission; delete the state file to start over (abandoning the "
                f"in-flight batch) or restore the original inputs."
            )
    else:
        if record:
            raise RuntimeError(
                f"Saved batch for turn {turn_index} was submitted to "
                f"'{record['provider']}' but this run is configured for "
                f"'{ctx.provider.name}'. Results cannot be mixed across providers; "
                f"use a separate batch_state_dir per provider."
            )
        bodies = [
            build_body(ctx.provider_name, messages, generation_params,
                       model_name, ctx.max_output_tokens)
            for messages in inputs
        ]
        job_id = ctx.provider.submit(bodies)
        n_requests = len(bodies)
        ctx.state.record(turn_index, job_id, ctx.provider.name, n_requests)

    waited = 0
    while True:
        terminal, status = ctx.provider.poll(job_id)
        if terminal:
            print(f"  Batch finished after {waited}s (status: {status}).", flush=True)
            break
        print(
            f"  Waiting on batch ({status}); {waited}s elapsed.",
            flush=True,
        )
        time.sleep(ctx.poll_interval_secs)
        waited += ctx.poll_interval_secs

    results = ctx.provider.fetch(job_id, n_requests)
    ctx.state.mark_complete(turn_index)

    failures: dict[str, int] = {}
    for result in results:
        if result.failed:
            reason = result.error or "empty response"
            failures[reason] = failures.get(reason, 0) + 1
    if failures:
        summary = ", ".join(f"{count}x {reason}" for reason, count in sorted(failures.items()))
        print(f"WARNING: {sum(failures.values())}/{len(results)} requests failed: {summary}")

    return (
        [r.text for r in results],
        sum(r.prompt_tokens for r in results),
        sum(r.completion_tokens for r in results),
        sum(1 for r in results if r.failed),
    )


def _build_messages(prompt: Prompt, turn_index: int, responses: list[str]) -> list[dict]:
    """Build an OpenAI-compatible message list for a single prompt's chat turn.

    Constructs user/assistant pairs for turns 0..turn_index. If use_multiturn
    is False, only the final user message is returned.

    {{RESP_N}} tokens in each turn are replaced with the response from turn N.

    Args:
        prompt: The prompt to build messages for.
        turn_index: The current turn index.
        responses: A list of responses from previous turns.

    Returns:
        A list of message dicts suitable for the OpenAI API.
    """
    messages: list[dict] = []

    if turn_index == 0 or prompt.use_multiturn:
        for ex_user, ex_asst in prompt.examples:
            messages.append({"role": "user", "content": ex_user})
            messages.append({"role": "assistant", "content": ex_asst})

    chat_messages: list[dict] = []
    for t in range(turn_index + 1):
        text = prompt.chat_turns[t]
        for i in range(t):
            text = text.replace(f"{{{{RESP_{i}}}}}", responses[i])
        chat_messages.append({"role": "user", "content": text})
        if t < turn_index:
            # A failed earlier turn leaves "" here. Anthropic rejects empty text
            # blocks outright, and every provider would be billed for a
            # generation conditioned on nothing, so build_dataset drops such
            # rows before this point. Fail loudly if one slips through.
            if not responses[t].strip():
                raise ValueError(
                    f"Turn {t} response is empty while building turn {turn_index}. "
                    f"Rows whose earlier turn failed must be excluded from later turns."
                )
            chat_messages.append({"role": "assistant", "content": responses[t]})

    if not prompt.use_multiturn:
        chat_messages = [chat_messages[-1]]

    messages.extend(chat_messages)
    return messages


def build_dataset(
    samples: list[str],
    api_url: str,
    prompts: PromptSet,
    generation_params: dict[str, Any],
    api_key: str = "EMPTY",
    model_name: str = "",
    batch_ctx: "BatchContext | None" = None,
) -> tuple[dict[str, list[Any]], int, int, int]:
    """Iteratively build a dataset dict by batching across the prompt dimension.

    For each chat turn:
    1. Collect the messages for every sample still eligible at this index.
    2. Dispatch them - concurrently against a live endpoint, or as one offline
       batch job when *batch_ctx* is given.
    3. Scatter the responses back into a per-sample column (response_N).
    4. Append the response to each sample's response history for the next turn.

    A sample is eligible at a turn when its prompt has a turn at that index
    *and* none of its earlier turns came back empty. Dropping rows with a
    failed ancestor keeps invalid conversation histories (an empty assistant
    message) out of later requests and avoids paying to condition a generation
    on nothing.

    Some rows will have empty strings in response_N columns if their prompt
    has fewer chat turns than the maximum, or if an earlier turn failed.

    Special tokens replaced in prompts:
    - {{DOC}} is replaced with the sample text (done in PromptSet.map).
    - {{RESP_N}} is replaced with the Nth response.

    Args:
        samples: List of text samples to process.
        api_url: OpenAI-compatible API base URL.
        prompts: PromptSet to draw prompts from.
        generation_params: Overrides for sampling params.
        api_key: API key for the endpoint.
        model_name: Model name for the API call.
        batch_ctx: When given, dispatch each turn through the provider's
            offline batch API instead of the live endpoint. ``None`` (the
            default) keeps the synchronous path.

    Returns:
        A tuple of (dataset_columns_dict, total_prompt_tokens, total_completion_tokens, total_failed_requests).
        The dict has keys: "original", "prompt", "response_0", "response_1",
        ..., "final_response".
    """
    print(f"Processing {len(samples)} samples...")

    if not samples:
        print("No samples to process; returning an empty dataset.")
        return {"original": [], "prompt": [], "final_response": []}, 0, 0, 0

    mapped_prompts, prompt_labels = prompts.map(samples)
    max_turns = max(len(p.chat_turns) for p in mapped_prompts)
    responses_grouped: list[list[str]] = [[] for _ in samples]
    dataset_columns: dict[str, list] = {"original": samples, "prompt": prompt_labels}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_failed_requests = 0

    for turn_idx in range(max_turns):
        print(f"Processing chat turn {turn_idx} / {max_turns - 1}...")
        batch_inputs: list[list[dict]] = []
        active_indices: list[int] = []

        for sample_idx, prompt in enumerate(mapped_prompts):
            prior_responses = responses_grouped[sample_idx]
            if turn_idx >= len(prompt.chat_turns):
                continue
            # A row whose earlier turn came back empty cannot continue: its
            # history would carry an empty assistant message, which Anthropic
            # rejects and which every provider bills for while producing
            # garbage that post-processing discards anyway.
            if any(not response.strip() for response in prior_responses):
                continue
            messages = _build_messages(prompt, turn_idx, prior_responses)
            batch_inputs.append(messages)
            active_indices.append(sample_idx)

        if batch_ctx is None:
            batch_responses, p_tokens, c_tokens, f_reqs = batch_generate(
                api_url, batch_inputs, generation_params, api_key, model_name
            )
        else:
            batch_responses, p_tokens, c_tokens, f_reqs = batch_generate_offline(
                batch_ctx, batch_inputs, generation_params, model_name, turn_idx
            )
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        total_failed_requests += f_reqs

        turn_responses = [""] * len(samples)
        for i, sample_idx in enumerate(active_indices):
            turn_responses[sample_idx] = batch_responses[i]
            responses_grouped[sample_idx].append(batch_responses[i])

        col_name = f"response_{turn_idx}"
        dataset_columns[col_name] = turn_responses
        print(f"  -> Column '{col_name}' created with {len(active_indices)} active responses.")

    dataset_columns["final_response"] = [responses[-1] if responses else "" for responses in responses_grouped]

    print(f"Dataset dict built with {len(samples)} rows and {len(dataset_columns)} columns.")

    return dataset_columns, total_prompt_tokens, total_completion_tokens, total_failed_requests
