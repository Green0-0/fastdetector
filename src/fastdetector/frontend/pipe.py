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
    source_column: str,
    source_dataset_name: str,
    batch_id: int | None = None,
    num_samples: int | None = None,
) -> tuple[Dataset, str]:
    """Run the generation pipeline and return the result dataset.

    Args:
        globals_config: GlobalsConfig with venv paths, dataset name resolution, etc.
        pipe_config: Pipeline settings (engine, model_name, etc).
        prompt_file: Path to the prompt file.
        source_column: The dataset column to extract text from.
        source_dataset_name: Dataset to read samples from.
        batch_id: Optional shard index (used for both the source subset_index
            and the target config_name).
        num_samples: Number of samples to process. ``None`` (the default)
            processes every row of the shard, which is what the pipeline
            entrypoints do - how much data a run covers is decided once, when
            the source dataset is sharded (scripts/shard_dataset.py).

    Returns:
        A tuple of (Dataset, readme_content) containing the generated samples and metadata.
    """
    start_time = time.time()
    engine = pipe_config.engine

    print("Running generation pipeline...")
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

    ALL_SAMPLING_PARAMS = [
        "temperature", "top_p", "top_k", "presence_penalty", "disable_thinking",
        "top_a", "xtc_probability", "nsigma",
    ]
    ignored_params = {
        p: getattr(pipe_config, p)
        for p in ALL_SAMPLING_PARAMS
        if getattr(pipe_config, p, None) is not None and p not in engine.valid_sampling_params
    }
    if ignored_params:
        print(
            f"WARNING: The following sampling params are set in the config but "
            f"are not supported by the {engine.value} engine and will be "
            f"IGNORED: {ignored_params}"
        )

    if extra_body:
        generation_params["extra_body"] = extra_body

    print(f"Loading prompts from file: {os.path.basename(prompt_file)}")
    prompt_list = load_prompts([prompt_file])
    prompts = PromptSet(prompt_list)

    sample_target = "all" if num_samples is None else num_samples
    print(f"Streaming {sample_target} samples from {source_dataset_name} (subset index {batch_id})...")
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
        if num_samples is not None and len(samples) >= num_samples:
            break

    print(f"Dropped {dropped_count} samples over the length limit ({length_limit}).")
    print(f"Loaded {len(samples)} samples with a total of {tokens_or_words_processed} {unit}.")

    if engine.is_local_server:
        venv_path = globals_config.vllm_venv_path if engine == EngineConfig.VLLM else globals_config.aphrodite_venv_path
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
- Sampling Params (as sent to the engine): {json.dumps(generation_params)}
- Ignored Params (unsupported by this engine): {json.dumps(ignored_params) if ignored_params else "None"}

- Prompt File: {prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {source_dataset_name}
- Source Column: {source_column}
- Target Num Samples: {sample_target}
- Dropped Samples (over length limit {length_limit} {unit}): {dropped_count}
- Failed API Requests: {failed_requests}

- Total Input Tokens Processed: {prompt_tokens}
- Total Output Tokens Processed: {completion_tokens}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: {engine.value}
"""
    print("Done!")
    return result_ds, readme_content