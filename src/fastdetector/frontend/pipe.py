import os
import time
import json
from transformers import AutoTokenizer
from datasets import Dataset
from fastdetector.utils import load_dataset_auto_shard
from fastdetector.prompting.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context
from fastdetector.frontend.toml_config import GlobalsConfig, PipeConfig
from fastdetector.frontend.engine_config import EngineConfig


def run_pipeline(
    globals_config: GlobalsConfig,
    pipe_config: PipeConfig,
    prompt_file: str,
    num_samples: int,
    source_column: str,
    source_dataset_name: str,
    batch_id: int | None = None,
) -> tuple[Dataset, str]:
    """Run the generation pipeline and return the result dataset.

    Args:
        globals_config: GlobalsConfig with venv paths, dataset name resolution, etc.
        pipe_config: Pipeline settings (engine, model_name, etc).
        prompt_file: Path to the prompt file.
        num_samples: Number of samples to process.
        source_column: The dataset column to extract text from.
        source_dataset_name: Dataset to read samples from.
        batch_id: Optional shard index (used for both the source subset_index
            and the target config_name).

    Returns:
        A tuple of (Dataset, readme_content) containing the generated samples and metadata.
    """
    start_time = time.time()
    engine = pipe_config.engine

    print(f"Running generation pipeline...")
    print(f"Using engine: {engine.value}")
    print(f"Using model: {pipe_config.model_name}")
    
    generation_params = {}
    extra_body = {}
    for param in engine.valid_sampling_params:
        val = getattr(pipe_config, param, None)
        if val is None:
            continue
        if param == "disable_thinking":
            if val:
                if engine.is_proprietary:
                    generation_params["reasoning_effort"] = "none"
                else:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif param in {"temperature", "top_p", "presence_penalty"}:
            generation_params[param] = val
        else:
            extra_body[param] = val

    if extra_body:
        generation_params["extra_body"] = extra_body

    print(f"Loading prompts from file: {os.path.basename(prompt_file)}")
    prompt_list = load_prompts([prompt_file])
    prompts = PromptSet(prompt_list)

    print(f"Streaming {num_samples} samples from {source_dataset_name} (subset index {batch_id})...")
    ds = load_dataset_auto_shard(source_dataset_name, split="train", subset_index=batch_id)

    samples = []
    tokens_or_words_processed = 0
    dropped_count = 0

    tokenizer = None
    if not engine.is_proprietary:
        tokenizer = AutoTokenizer.from_pretrained(pipe_config.model_name)

    length_limit = pipe_config.max_input_len
    unit = "tokens" if tokenizer is not None else "words"

    for row in ds:
        text = row[source_column]
        text = str(text) if text is not None else ""

        if tokenizer is not None:
            count = len(tokenizer.encode(text))
        else:
            count = len(text.split())

        if length_limit is not None and count > length_limit:
            dropped_count += 1
            continue

        samples.append(text)
        tokens_or_words_processed += count
        if len(samples) >= num_samples:
            break

    print(f"Dropped {dropped_count} samples over the length limit ({length_limit}).")
    print(f"Loaded {len(samples)} samples with a total of {tokens_or_words_processed} {unit}.")

    if engine.is_local_server:
        venv_path = globals_config.vllm_venv_path if engine == EngineConfig.VLLM else globals_config.aphrodite_venv_path
        # Only forward max_model_len when set; passing None would override the
        # server default and end up as a literal "--max-model-len None" flag.
        server_kwargs = {}
        if pipe_config.max_model_len is not None:
            server_kwargs["max_model_len"] = pipe_config.max_model_len
        with llm_server_context(
            engine=engine,
            model_name=pipe_config.model_name,
            venv_path=venv_path,
            parallelization_type=pipe_config.parallelization_type,
            port=None,
            max_num_seqs=pipe_config.max_num_seqs,
            max_num_batched_tokens=pipe_config.max_num_batched_tokens,
            **server_kwargs,
        ) as api_url:
            print(f"Using API endpoint: {api_url}")
            result_dict, prompt_tokens, completion_tokens, failed_requests = build_dataset(
                samples,
                api_url=api_url,
                prompts=prompts,
                generation_params=generation_params,
            )
    else:
        if pipe_config.api_key_env is not None:
            api_key = os.environ.get(pipe_config.api_key_env, "EMPTY")
        else:
            api_key = "EMPTY"
        print(f"Using API endpoint: {pipe_config.api_url}")
        result_dict, prompt_tokens, completion_tokens, failed_requests = build_dataset(
            samples,
            api_url=pipe_config.api_url,
            prompts=prompts,
            generation_params=generation_params,
            api_key=api_key,
            model_name=pipe_config.model_name,
        )

    if result_dict:
        col_lengths = {k: len(v) for k, v in result_dict.items()}
        unique_lengths = set(col_lengths.values())
        if len(unique_lengths) > 1:
            raise RuntimeError(
                f"build_dataset returned columns of mismatched lengths: {col_lengths}"
            )
        num_rows = unique_lengths.pop()
    else:
        num_rows = 0

    result_dict["generator_model"] = [pipe_config.model_name] * num_rows
    result_dict["generation_params"] = [json.dumps(generation_params)] * num_rows

    result_ds = Dataset.from_dict(result_dict)

    total_runtime = time.time() - start_time
    readme_content = f"""# Auto-Generated FastDetector Dataset
- Model Name: {pipe_config.model_name}
- Temperature: {pipe_config.temperature}
- Top P: {pipe_config.top_p}
- Top K: {pipe_config.top_k}
- Presence Penalty: {pipe_config.presence_penalty}
- Disable Thinking: {pipe_config.disable_thinking}

- Prompt File: {prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {source_dataset_name}
- Source Column: {source_column}
- Target Num Samples: {num_samples}
- Dropped Samples (over length limit {length_limit} {unit}): {dropped_count}
- Failed API Requests: {failed_requests}

- Total Input Tokens Processed: {prompt_tokens}
- Total Output Tokens Processed: {completion_tokens}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: {engine.value}
"""
    print("Done!")
    return result_ds, readme_content