import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset, concatenate_datasets
from fastdetector.prompts import Prompt, PromptSet

MAX_RETRIES = 5

MAX_CONCURRENT = 256

async def _send_request(client: AsyncOpenAI, semaphore: asyncio.Semaphore, messages: list[dict]) -> str:
    """Send a single chat completion request. Retries are handled by the OpenAI client."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="",
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Request failed: {e}")
            return ""

async def _batch_generate_async(api_url: str, inputs: list[list[dict]]) -> list[str]:
    """Fires requests concurrently with a bounded semaphore."""
    client = AsyncOpenAI(
        base_url=api_url,
        api_key="EMPTY",
        max_retries=MAX_RETRIES,
        timeout=180.0,
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(inputs)
    completed = 0

    async def _tracked_request(messages: list[dict]) -> str:
        nonlocal completed
        result = await _send_request(client, semaphore, messages)
        completed += 1
        if completed % 1000 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} requests complete")
        return result

    tasks = [_tracked_request(messages) for messages in inputs]
    results = await asyncio.gather(*tasks)
    await client.close()
    return list(results)

def batch_generate(api_url: str, inputs: list[list[dict]]) -> list[str]:
    """
    Takes in a list of OpenAI compatible chat conversations, and returns a list of responses.
    Uses the async OpenAI client to fire all requests concurrently via asyncio.gather.

    Args:
        api_url: The URL of the OpenAI-compatible chat completions endpoint
                 (e.g. "http://localhost:8000/v1").
        inputs: A list of conversations, where each conversation is a list of
                {"role": ..., "content": ...} message dicts.

    Returns:
        A list of assistant response strings, one per input conversation.
    """
    return asyncio.run(_batch_generate_async(api_url, inputs))

def _build_messages(prompt: Prompt, turn_index: int, responses: list[str]) -> list[dict]:
    """
    Build an OpenAI-compatible message list for a given chat turn.

    Constructs user/assistant pairs for turns 0..turn_index. If use_multiturn
    is False, only the final user message is returned.

    {{RES_N}} tokens in each turn are replaced with the response from turn N.
    """
    messages: list[dict] = []
    for t in range(turn_index + 1):
        text = prompt.chat_turns[t]
        for i in range(t):
            text = text.replace(f"{{{{RES_{i}}}}}", responses[i])
        messages.append({"role": "user", "content": text})
        if t < turn_index:
            messages.append({"role": "assistant", "content": responses[t]})

    if not prompt.use_multiturn:
        return [messages[-1]]
    return messages

def build_dataset(
    samples: list[str],
    target: str,
    api_url: str,
    prompts: PromptSet,
    append: bool,
    use_test: bool = False,
):
    """
    Iteratively builds the dataset by batching across the prompt dimension.

    It creates a batch along the dataset dimension (with one prompt mapped to each row), calling the batch generator, and then
    creates a new column each time the batch generator returns. A column is also created for the
    original sample.

    Then, a batch is created for anything with a second chat turn in the prompts, and this process
    is repeated until no more chat turns are present in any prompt (some rows will have columns
    with empty strings).

    These special tokens are replaced in the prompt:

    {{DOC}} is replaced with the sample text.
    {{RES_N}} is replaced with the Nth response. Note that when multiturn is true, the entire
    chat history is used, otherwise only the latest message is used.

    Args:
        samples: List of text samples to process.
        target: HuggingFace repo to push results to.
        api_url: OpenAI-compatible API base URL.
        prompts: PromptSet to draw prompts from.
        append: If True, append to any existing dataset at target.
        use_test: If True, use test set prompts instead of train set.
    """
    print(f"Processing {len(samples)} samples...")

    # Map one prompt to each sample (replaces {{DOC}})
    mapped_prompts, prompt_labels = prompts.map(samples, use_test=use_test)

    # Determine the maximum number of chat turns across all mapped prompts
    max_turns = max(len(p.chat_turns) for p in mapped_prompts)

    # Storage for all responses: responses_by_turn[turn_idx][sample_idx] = response string
    responses_by_turn: list[list[str]] = []

    # Build the dataset columns, starting with the original text and prompt template
    columns: dict[str, list[str]] = {"original": samples, "prompt": prompt_labels}

    for turn_idx in range(max_turns):
        print(f"Processing chat turn {turn_idx} / {max_turns - 1}...")

        # Build message lists for each sample at this turn
        batch_inputs: list[list[dict]] = []
        active_indices: list[int] = []

        for sample_idx, prompt in enumerate(mapped_prompts):
            if turn_idx < len(prompt.chat_turns):
                prior_responses = [responses_by_turn[t][sample_idx] for t in range(turn_idx)]
                messages = _build_messages(prompt, turn_idx, prior_responses)
                batch_inputs.append(messages)
                active_indices.append(sample_idx)

        # Call the batch generator
        batch_responses = batch_generate(api_url, batch_inputs)

        # Distribute responses back to the full sample list
        turn_responses = [""] * len(samples)
        for i, sample_idx in enumerate(active_indices):
            turn_responses[sample_idx] = batch_responses[i]

        responses_by_turn.append(turn_responses)

        # Create a column for this turn
        col_name = f"response_{turn_idx}"
        columns[col_name] = turn_responses
        print(f"  -> Column '{col_name}' created with {len(active_indices)} active responses.")

    # Build the final HuggingFace dataset
    result_ds = Dataset.from_dict(columns)

    if append:
        try:
            existing_ds = load_dataset(target, split="train")
            result_ds = concatenate_datasets([existing_ds, result_ds])
            print(f"Appended to existing dataset '{target}'.")
        except Exception:
            print(f"No existing dataset found at '{target}', creating new.")

    result_ds.push_to_hub(target)
    print(f"Dataset pushed to '{target}' with {len(result_ds)} rows and {len(columns)} columns.")
