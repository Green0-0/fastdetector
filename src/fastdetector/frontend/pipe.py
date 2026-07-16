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
from fastdetector.frontend.config import GenConfig, GlobalsConfig

def run_pipeline(
    gen_config: GenConfig,
    globals_config: GlobalsConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    batch_id: int | None = None
):
    start_time = time.time()
    pipe_cfg = gen_config.pipeline

    print(f"Running generation pipeline...")
    print(f"Loading configuration from: {gen_config.config_path}")
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

    if pipe_cfg.engine in ["vllm", "aphrodite"]:
        tokenizer = AutoTokenizer.from_pretrained(pipe_cfg.model_name)
        for row in ds:
            text = row[gen_config.source_column]
            text = str(text) if text is not None else ""
            num_tokens = len(tokenizer.encode(text))
            
            if pipe_cfg.max_input_tokens is not None:
                if num_tokens > pipe_cfg.max_input_tokens:
                    dropped_count += 1
                    continue
                    
            samples.append(text)
            tokens_or_words_processed += num_tokens
            if len(samples) >= gen_config.num_samples:
                break
    else:
        def get_len(text): return len(text.split())
        for row in ds:
            text = row[gen_config.source_column]
            text = str(text) if text is not None else ""
            num_words = get_len(text)
            
            if pipe_cfg.max_dataset_words is not None:
                if num_words > pipe_cfg.max_dataset_words:
                    dropped_count += 1
                    continue
                    
            samples.append(text)
            tokens_or_words_processed += num_words
            if len(samples) >= gen_config.num_samples:
                break
    print(f"Dropped {dropped_count} samples over the length limit ({pipe_cfg.max_input_tokens if pipe_cfg.engine in ["vllm", "aphrodite"] else pipe_cfg.max_dataset_words}).")
    print(f"Loaded {len(samples)} samples with a total of {tokens_or_words_processed} tokens.")

    if pipe_cfg.engine in ["vllm", "aphrodite"]:
        with llm_server_context(engine=pipe_cfg.engine, model_name=pipe_cfg.model_name, port=None, max_model_len=pipe_cfg.max_model_len) as api_url:
            print(f"Using API endpoint: {api_url}")
            result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
                samples=samples,
                api_url=api_url,
                prompts=prompts,
                generation_params=generation_params,
            )
    else:
        api_key = os.environ.get(pipe_cfg.api_key_env, "EMPTY") if pipe_cfg.api_key_env else "EMPTY"
        print(f"Using API endpoint: {pipe_cfg.api_url}")
        result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
            samples=samples,
            api_url=pipe_cfg.api_url,
            prompts=prompts,
            generation_params=generation_params,
            api_key=api_key,
            model_name=pipe_cfg.model_name,
        )
        
    num_rows = len(next(iter(result_dict.values()))) if result_dict else 0
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
