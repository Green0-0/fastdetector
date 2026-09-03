import asyncio
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from fastdetector.frontend.toml_config import PipeConfig
from fastdetector.generation_checkpoint import GenerationCheckpoint, TurnLog
from fastdetector.prompting.prompts import Prompt, PromptSet
from fastdetector.providers import BatchProvider, BatchState

@dataclass(frozen=True)
class _RequestResult:
    """One online request result with an unambiguous success state."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.text.strip())


async def _send_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages: list[dict[str, str]],
    generation_params: dict[str, Any],
    model_name: str = "",
) -> _RequestResult:
    """Send a single chat completion request.

    Args:
        client: The AsyncOpenAI client.
        semaphore: Bounded semaphore limiting concurrency.
        messages: OpenAI-compatible message list.
        generation_params: Sampling param overrides.
        model_name: Model name for the API call.

    Returns:
        Structured response with explicit error state.
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
                return _RequestResult("", 0, 0, "response contained no choices")

            usage = getattr(response, "usage", None)
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            text = response.choices[0].message.content or ""
            if not text.strip():
                return _RequestResult(
                    text, prompt_tokens, completion_tokens, "response contained no usable text"
                )
            return _RequestResult(text, prompt_tokens, completion_tokens)
        except Exception as e:
            print(f"Request failed: {e}")
            return _RequestResult("", 0, 0, str(e))


async def _batch_generate_async(
    api_url: str,
    inputs: list[list[dict[str, str]]],
    generation_params: dict[str, Any],
    api_key: str = "EMPTY",
    model_name: str = "",
    row_ids: list[int] | None = None,
    turn_log: TurnLog | None = None,
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
        row_ids: Stable sample indices aligned with ``inputs``. Required with
            ``turn_log``.
        turn_log: Optional durable log which records each successful response.

    Returns:
        A tuple of (list of assistant response strings, total prompt tokens,
        total completion tokens, failed request count). Failed requests appear as empty strings.
    """
    if (row_ids is None) != (turn_log is None):
        raise ValueError("row_ids and turn_log must be provided together")
    stable_ids = list(range(len(inputs))) if row_ids is None else row_ids
    if len(stable_ids) != len(inputs):
        raise ValueError("row_ids must have the same length as inputs")
    if not inputs:
        return [], 0, 0, 0

    client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        max_retries=5,
        timeout=360.0,
    )
    semaphore = asyncio.Semaphore(256)
    total = len(inputs)
    completed = 0
    failed_count = 0

    try:
        async def _tracked_request(
            row_id: int, messages: list[dict[str, str]]
        ) -> _RequestResult:
            """Send a request and update progress and failure counts.

            Args:
                messages: List of message dicts.

            Returns:
                Structured result.
            """
            nonlocal completed, failed_count
            result = await _send_request(client, semaphore, messages, generation_params, model_name=model_name)
            completed += 1
            if not result.succeeded:
                failed_count += 1
            elif turn_log is not None:
                turn_log.append(
                    row_id, result.text, result.prompt_tokens, result.completion_tokens
                )
            if completed % 100 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} requests complete", flush=True)
            return result

        tasks = [
            _tracked_request(row_id, messages)
            for row_id, messages in zip(stable_ids, inputs)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await client.close()

    texts = [result.text for result in results]
    prompt_tokens = sum(result.prompt_tokens for result in results)
    completion_tokens = sum(result.completion_tokens for result in results)

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
    row_ids: list[int] | None = None,
    turn_log: TurnLog | None = None,
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
        row_ids: Stable sample indices aligned with ``inputs``. Required with
            ``turn_log``.
        turn_log: Optional durable log which records completed requests.

    Returns:
        A tuple of (list of assistant response strings, total prompt tokens,
        total completion tokens, failed request count). Failed requests appear as empty strings.
    """
    return asyncio.run(
        _batch_generate_async(
            api_url, inputs, generation_params, api_key, model_name, row_ids, turn_log
        )
    )


def batch_generate_offline(
    provider: BatchProvider,
    state: BatchState,
    config: PipeConfig,
    inputs: list[list[dict[str, str]]],
    generation_params: dict[str, Any],
    model_name: str,
    turn_index: int,
) -> tuple[list[str], int, int, int]:
    """Run one chat turn through a provider's offline batch API. 
    
    A run that already has a saved job for this turn resumes polling rather than submitting again.

    Args:
        provider: The transport to submit through.
        state: Durable record of submitted jobs.
        config: Pipeline settings supplying the payload dialect, output cap,
            and poll interval.
        inputs: Conversations for the rows active at this turn.
        generation_params: Params already filtered to this provider's accepted set.
        model_name: Model ID, or a deployment name.
        turn_index: Chat turn index, used as the state key.

    Returns:
        A tuple of (response strings, total prompt tokens, total completion
        tokens, failed request count).
    """
    if not inputs:
        return [], 0, 0, 0

    record = state.get(turn_index)
    if record and record["provider"] == provider.name:
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
                f"'{provider.name}'. Results cannot be mixed across providers; "
                f"use a separate batch_state_dir per provider."
            )
        job_id = provider.submit(inputs, generation_params, model_name, config.max_output_tokens)
        n_requests = len(inputs)
        state.record(turn_index, job_id, provider.name, n_requests)

    waited = 0
    while True:
        terminal, status = provider.poll(job_id)
        if terminal:
            print(f"  Batch finished after {waited}s (status: {status}).", flush=True)
            break
        print(f"  Waiting on batch ({status}); {waited}s elapsed.", flush=True)
        time.sleep(config.batch_poll_interval_secs)
        waited += config.batch_poll_interval_secs

    results = provider.fetch(job_id, n_requests)
    state.mark_complete(turn_index)

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
    provider: BatchProvider | None = None,
    state: BatchState | None = None,
    config: PipeConfig | None = None,
    checkpoint: GenerationCheckpoint | None = None,
    max_failure_rate: float | None = None,
) -> tuple[dict[str, list[Any]], int, int, int]:
    """Iteratively build a dataset dict by batching across the prompt dimension.

    For each chat turn:
    1. Collect the messages for every sample still eligible at this index.
    2. Dispatch them - concurrently against a live endpoint, or as one offline
       batch job when *provider* is given.
    3. Scatter the responses back into a per-sample column (response_N).
    4. Append the response to each sample's response history for the next turn.

    A sample is eligible at a turn when its prompt has a turn at that index
    and none of its earlier turns came back empty.

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
        provider: When given, dispatch each turn through this provider's
            offline batch API instead of the live endpoint.
        state: Durable record of submitted jobs. Required with *provider*.
        config: Pipeline settings for the batch path. Required with *provider*.
        checkpoint: Prepared online-generation checkpoint. Not used with an
            offline batch provider.
        max_failure_rate: Abort when failures exceed this fraction of the
            active rows in any online turn. ``None`` disables the guard.

    Returns:
        A tuple of (dataset_columns_dict, total_prompt_tokens, total_completion_tokens, total_failed_requests).
        The dict has keys: "original", "prompt", "response_0", "response_1",
        ..., "final_response".

    Raises:
        ValueError: if only some of provider/state/config are given.
    """
    batch_args = (provider, state, config)
    if any(a is not None for a in batch_args) and not all(a is not None for a in batch_args):
        raise ValueError(
            "provider, state, and config must be given together to use the "
            "offline batch path."
        )
    if provider is not None and checkpoint is not None:
        raise ValueError("online generation checkpoints cannot be combined with an offline provider")
    if max_failure_rate is not None and not 0.0 <= max_failure_rate <= 1.0:
        raise ValueError("max_failure_rate must be between 0 and 1")
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
            if any(not response.strip() for response in prior_responses):
                continue
            messages = _build_messages(prompt, turn_idx, prior_responses)
            batch_inputs.append(messages)
            active_indices.append(sample_idx)

        active_count = len(batch_inputs)
        attempted_count = active_count
        if provider is None:
            if checkpoint is None:
                batch_responses, p_tokens, c_tokens, f_reqs = batch_generate(
                    api_url, batch_inputs, generation_params, api_key, model_name
                )
            else:
                turn_log = checkpoint.turn(turn_idx)
                cached = {
                    row_id: saved
                    for row_id in active_indices
                    if (saved := turn_log.get(row_id)) is not None
                }
                pending = [
                    (row_id, messages)
                    for row_id, messages in zip(active_indices, batch_inputs)
                    if row_id not in cached
                ]
                attempted_count = len(pending)
                if cached:
                    print(f"  Restored {len(cached)} responses from checkpoint.")
                generated, p_tokens, c_tokens, f_reqs = batch_generate(
                    api_url,
                    [messages for _, messages in pending],
                    generation_params,
                    api_key,
                    model_name,
                    row_ids=[row_id for row_id, _ in pending],
                    turn_log=turn_log,
                )
                generated_by_id = dict(zip((row_id for row_id, _ in pending), generated))
                batch_responses = [
                    cached[row_id].text if row_id in cached else generated_by_id[row_id]
                    for row_id in active_indices
                ]
                p_tokens += sum(record.prompt_tokens for record in cached.values())
                c_tokens += sum(record.completion_tokens for record in cached.values())
        else:
            assert state is not None and config is not None  # checked on entry
            batch_responses, p_tokens, c_tokens, f_reqs = batch_generate_offline(
                provider, state, config, batch_inputs, generation_params,
                model_name, turn_idx
            )
        if (
            provider is None
            and max_failure_rate is not None
            and active_count
            and f_reqs / active_count > max_failure_rate
        ):
            if checkpoint is not None:
                checkpoint.flush()
            raise RuntimeError(
                f"Generation engine appears unhealthy on turn {turn_idx}: "
                f"{f_reqs}/{attempted_count} newly attempted requests failed "
                f"({f_reqs / active_count:.1%} of {active_count} active rows), "
                f"above the configured "
                f"{max_failure_rate:.1%} limit. The checkpoint was retained."
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
