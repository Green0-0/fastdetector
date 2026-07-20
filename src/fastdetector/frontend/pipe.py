import os
import time
import json
from transformers import AutoTokenizer
from datasets import Dataset
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.prompting.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.utils import upload_dataset, upload_readme
from fastdetector.llm_utils import llm_server_context
from fastdetector.frontend.config import GenConfig, FilterConfig, GlobalsConfig

# run_pipeline is shared by gen.py and filter.py; both config types expose
# the same .pipeline / .source_column / .num_samples / .prompt_file interface.
PipelineConfigT = GenConfig | FilterConfig

def run_pipeline(
    gen_config: PipelineConfigT,
    globals_config: GlobalsConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    batch_id: int | None = None,
) -> Dataset:
    """Run the generation pipeline and upload the result dataset.

    Loads samples from *source_dataset_name*, generates responses via the
    configured LLM engine, builds a result dataset, uploads it to
    *target_dataset_name*, and returns the in-memory Dataset so callers
    (e.g. filter.py) can continue processing without re-loading from disk/Hub.

    Args:
        gen_config: GenConfig or FilterConfig (both expose .pipeline,
            .source_column, .num_samples, .prompt_file).
        globals_config: GlobalsConfig with cache_dir, save_locally_instead, etc.
        source_dataset_name: Dataset to read samples from.
        target_dataset_name: Dataset to upload the result to.
        batch_id: Optional shard index (used for both the source subset_index
            and the target config_name).

    Returns:
        The in-memory Dataset that was uploaded. Callers that don't need
        the in-memory copy can ignore the return value.
    """
    start_time = time.time()
    pipe_cfg = gen_config.pipeline

    print(f"Running generation pipeline...")
    print(f"Target dataset: {target_dataset_name}")
    print(f"Using engine: {pipe_cfg.engine}")
    print(f"Using model: {pipe_cfg.model_name}")

    generation_params = {
        "temperature": pipe_cfg.temperature,
        "top_p": pipe_cfg.top_p,
        "top_k": pipe_cfg.top_k,
        "presence_penalty": pipe_cfg.presence_penalty,
        "disable_thinking": pipe_cfg.disable_thinking,
    }

    if pipe_cfg.engine == "oai":
        generation_params.update({"is_api_model": True})

    if pipe_cfg.engine == "aphrodite":
        generation_params.update({
            "top_a": pipe_cfg.top_a,
            "xtc_probability": pipe_cfg.xtc_probability,
            "nsigma": pipe_cfg.nsigma,
        })

    print(f"Loading prompts from file: {os.path.basename(gen_config.prompt_file)}")
    prompt_list = load_prompts([gen_config.prompt_file])
    prompts = PromptSet(prompt_list)

    subset_idx = batch_id if batch_id is not None else 0
    config_name = f"shard_{batch_id}" if batch_id is not None else "default"

    print(f"Streaming {gen_config.num_samples} samples from {source_dataset_name} (subset index {subset_idx})...")
    ds = load_dataset(source_dataset_name, split="train", cache_dir=globals_config.cache_dir, subset_index=subset_idx)

    samples = []
    tokens_or_words_processed = 0
    dropped_count = 0

    uses_tokenizer = pipe_cfg.engine in ("vllm", "aphrodite")
    if uses_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(pipe_cfg.model_name)
        length_limit = pipe_cfg.max_input_tokens
    else:
        length_limit = pipe_cfg.max_dataset_words

    for row in ds:
        text = row[gen_config.source_column]
        text = str(text) if text is not None else ""

        if uses_tokenizer:
            count = len(tokenizer.encode(text))
        else:
            count = len(text.split())

        if length_limit is not None and count > length_limit:
            dropped_count += 1
            continue

        samples.append(text)
        tokens_or_words_processed += count
        if len(samples) >= gen_config.num_samples:
            break

    print(f"Dropped {dropped_count} samples over the length limit ({length_limit}).")
    print(f"Loaded {len(samples)} samples with a total of {tokens_or_words_processed} tokens.")

    if uses_tokenizer:
        with llm_server_context(engine=pipe_cfg.engine, model_name=pipe_cfg.model_name, port=None, max_model_len=pipe_cfg.max_model_len) as api_url:
            print(f"Using API endpoint: {api_url}")
            result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
                samples=samples,
                api_url=api_url,
                prompts=prompts,
                generation_params=generation_params,
            )
    else:
        # api_key_env may be None for a local unauthenticated endpoint.
        if pipe_cfg.api_key_env is not None:
            api_key = os.environ.get(pipe_cfg.api_key_env, "EMPTY")
        else:
            api_key = "EMPTY"
        print(f"Using API endpoint: {pipe_cfg.api_url}")
        result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
            samples=samples,
            api_url=pipe_cfg.api_url,
            prompts=prompts,
            generation_params=generation_params,
            api_key=api_key,
            model_name=pipe_cfg.model_name,
        )

    # Guard against silent column-length mismatches in result_dict.
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

    result_dict["generator_model"] = [pipe_cfg.model_name] * num_rows
    result_dict["generation_params"] = [json.dumps(generation_params)] * num_rows

    result_ds = Dataset.from_dict(result_dict)

    total_runtime = time.time() - start_time
    readme_content = f"""# Auto-Generated FastDetector Dataset
- Model Name: {pipe_cfg.model_name}
- Temperature: {pipe_cfg.temperature}
- Top P: {pipe_cfg.top_p}
- Top K: {pipe_cfg.top_k}
- Presence Penalty: {pipe_cfg.presence_penalty}
- Disable Thinking: {pipe_cfg.disable_thinking}

- Prompt File: {gen_config.prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {source_dataset_name}
- Source Column: {gen_config.source_column}
- Target Num Samples: {gen_config.num_samples}

- Total Input Tokens Processed: {total_prompt_tokens}
- Total Output Tokens Processed: {total_completion_tokens}

- Target Dataset: {target_dataset_name}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: {pipe_cfg.engine}
"""
    upload_dataset(
        dataset=result_ds,
        dataset_name=target_dataset_name,
        save_locally_instead=globals_config.save_locally_instead,
        cache_dir=globals_config.cache_dir,
        config_name=config_name
    )
    if not globals_config.save_locally_instead:
        upload_readme(dataset_name=target_dataset_name, readme_content=readme_content)
    print("Done!")
    return result_ds
