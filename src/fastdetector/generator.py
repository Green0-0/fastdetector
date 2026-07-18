import asyncio
from openai import AsyncOpenAI
from fastdetector.prompting.prompts import Prompt, PromptSet

MAX_RETRIES = 5

MAX_CONCURRENT = 256


async def _send_request(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    generation_params: dict,
    model_name: str = "",
) -> tuple[str, int, int]:
    """Send a single chat completion request.

    Retries are handled by the OpenAI client (max_retries=MAX_RETRIES set on
    the client in _batch_generate_async). On non-retriable failure, returns
    ("", 0, 0) so that the caller can track the failure and continue.

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
            }

            if generation_params.get("is_api_model", False):
                if generation_params.get("disable_thinking", False):
                    kwargs["reasoning_effort"] = "none"
            else:
                kwargs["temperature"] = generation_params.get("temperature", 1)
                kwargs["top_p"] = generation_params.get("top_p", 1)
                kwargs["presence_penalty"] = generation_params.get("presence_penalty", 0)

                extra_body = {}
                if generation_params.get("disable_thinking", False):
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}

                for key in ["top_k", "top_a", "xtc_probability", "nsigma"]:
                    val = generation_params.get(key)
                    if val is not None and val > 0:
                        extra_body[key] = val

                if extra_body:
                    kwargs["extra_body"] = extra_body

            response = await client.chat.completions.create(**kwargs)

            # Guard against empty choices — some endpoints return no choices
            # on content-filter rejection or other server-side errors.
            if not response.choices:
                print(f"Request returned no choices (possibly content-filtered).")
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
    inputs: list[list[dict]],
    generation_params: dict,
    api_key: str = "EMPTY",
    model_name: str = "",
) -> tuple[list[str], int, int]:
    """Fire requests concurrently with a bounded semaphore.

    The AsyncOpenAI client is created here and closed in a finally block so
    it is always released even if gather raises.

    Failed requests (those returning ("", 0, 0)) are counted and reported at
    the end. The empty-string responses are kept in the output list at their
    original positions so that the caller (build_dataset) can align them with
    the input samples.
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
        async def _tracked_request(messages: list[dict]) -> tuple[str, int, int]:
            nonlocal completed, failed_count
            result = await _send_request(client, semaphore, messages, generation_params, model_name=model_name)
            completed += 1
            # _send_request returns ("", 0, 0) on failure (caught exception
            # or empty choices).
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

    return texts, prompt_tokens, completion_tokens


def batch_generate(
    api_url: str,
    inputs: list[list[dict]],
    generation_params: dict,
    api_key: str = "EMPTY",
    model_name: str = "",
) -> tuple[list[str], int, int]:
    """Send a batch of OpenAI-compatible chat conversations concurrently.

    Uses the async OpenAI client to fire all requests concurrently via
    asyncio.gather, bounded by MAX_CONCURRENT.

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
        total completion tokens). Failed requests appear as empty strings.
    """
    return asyncio.run(_batch_generate_async(api_url, inputs, generation_params, api_key, model_name))


def _build_messages(prompt: Prompt, turn_index: int, responses: list[str]) -> list[dict]:
    """Build an OpenAI-compatible message list for a single prompt's chat turn.

    Constructs user/assistant pairs for turns 0..turn_index. If use_multiturn
    is False, only the final user message is returned.

    {{RESP_N}} tokens in each turn are replaced with the response from turn N.
    """
    messages: list[dict] = []

    # Add few-shot examples if this is the first turn or multiturn is enabled
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
            chat_messages.append({"role": "assistant", "content": responses[t]})

    if not prompt.use_multiturn:
        chat_messages = [chat_messages[-1]]

    messages.extend(chat_messages)
    return messages


def build_dataset(
    samples: list[str],
    api_url: str,
    prompts: PromptSet,
    generation_params: dict,
    use_test: bool = False,
    api_key: str = "EMPTY",
    model_name: str = "",
) -> tuple[dict[str, list], int, int]:
    """Iteratively build a dataset dict by batching across the prompt dimension.

    For each chat turn:
    1. Collect the messages for every sample that has a turn at this index.
    2. Call batch_generate to fire all requests concurrently.
    3. Scatter the responses back into a per-sample column (response_N).
    4. Append the response to each sample's response history for the next turn.

    Some rows will have empty strings in response_N columns if their prompt
    has fewer chat turns than the maximum.

    Special tokens replaced in prompts:
    - {{DOC}} is replaced with the sample text (done in PromptSet.map).
    - {{RESP_N}} is replaced with the Nth response.

    Args:
        samples: List of text samples to process.
        api_url: OpenAI-compatible API base URL.
        prompts: PromptSet to draw prompts from.
        use_test: If True, use test set prompts instead of train set.
        generation_params: Overrides for sampling params.
        api_key: API key for the endpoint.
        model_name: Model name for the API call.

    Returns:
        A tuple of (dataset_columns_dict, total_prompt_tokens, total_completion_tokens).
        The dict has keys: "original", "prompt", "response_0", "response_1",
        ..., "final_response".
    """
    print(f"Processing {len(samples)} samples...")

    mapped_prompts, prompt_labels = prompts.map(samples, use_test=use_test)
    max_turns = max(len(p.chat_turns) for p in mapped_prompts)
    responses_grouped: list[list[str]] = [[] for _ in samples]
    dataset_columns: dict[str, list] = {"original": samples, "prompt": prompt_labels}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for turn_idx in range(max_turns):
        print(f"Processing chat turn {turn_idx} / {max_turns - 1}...")
        batch_inputs: list[list[dict]] = []
        active_indices: list[int] = []

        for sample_idx, prompt in enumerate(mapped_prompts):
            if turn_idx < len(prompt.chat_turns):
                prior_responses = responses_grouped[sample_idx]
                messages = _build_messages(prompt, turn_idx, prior_responses)
                batch_inputs.append(messages)
                active_indices.append(sample_idx)

        batch_responses, p_tokens, c_tokens = batch_generate(api_url, batch_inputs, generation_params, api_key, model_name)
        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens

        turn_responses = [""] * len(samples)
        for i, sample_idx in enumerate(active_indices):
            turn_responses[sample_idx] = batch_responses[i]
            responses_grouped[sample_idx].append(batch_responses[i])

        col_name = f"response_{turn_idx}"
        dataset_columns[col_name] = turn_responses
        print(f"  -> Column '{col_name}' created with {len(active_indices)} active responses.")

    dataset_columns["final_response"] = [responses[-1] if responses else "" for responses in responses_grouped]

    print(f"Dataset dict built with {len(samples)} rows and {len(dataset_columns)} columns.")

    return dataset_columns, total_prompt_tokens, total_completion_tokens
