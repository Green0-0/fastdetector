import argparse
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

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate LLM data using Aphrodite engine.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to launch.")
    parser.add_argument("--temperature", type=float, required=True, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, required=True, help="Generation top-p.")
    parser.add_argument("--top-k", type=int, default=-1, help="Generation top-k.")
    parser.add_argument("--presence-penalty", type=float, required=True, help="Generation presence penalty.")
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to the chat template.")

    parser.add_argument("--top-a", type=float, default=0.0, help="Generation top-a.")
    parser.add_argument("--xtc", type=float, default=0.0, help="Generation xtc-probability (uses default threshold of 0.1).")
    parser.add_argument("--nsigma", type=float, default=0.0, help="Generation nsigma.")

    parser.add_argument("--max-model-len", type=int, required=True, help="Max model length.")
    parser.add_argument("--max-dataset-len", type=int, required=True, help="Max acceptable input length from the dataset.")

    parser.add_argument("--prompt-file", type=str, required=True, help="Prompt JSON file.")

    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--source-column", type=str, required=True, help="Source column.")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to stream.")
    
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    args = parser.parse_args()

    generation_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "disable_thinking": args.disable_thinking,
        # aphrodite specific
        "top_a": args.top_a,
        "xtc_probability": args.xtc,
        "nsigma": args.nsigma,
    }

    print(f"Loading filtering prompt from file: {os.path.basename(args.prompt_file)}")
    prompt_list = load_prompts([args.prompt_file])
    prompts = PromptSet(prompt_list)

    print(f"Loading tokenizer and streaming {args.num_samples} samples from {args.source_dataset}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)
    samples = []
    tokens_processed = 0
    for row in ds:
        text = row[args.source_column]
        text = str(text) if text is not None else ""
        num_tokens = len(tokenizer.encode(text))
        if num_tokens <= args.max_dataset_len:
            samples.append(text)
            tokens_processed += num_tokens
        if len(samples) >= args.num_samples:
            break
    print(f"Loaded {len(samples)} samples.")

    with llm_server_context(engine="aphrodite", model_name=args.model_name, port=None, max_model_len=args.max_model_len) as api_url:
        print(f"Using API endpoint: {api_url}")
        result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
            samples=samples,
            api_url=api_url,
            prompts=prompts,
            generation_params=generation_params,
        )
        
        num_rows = len(next(iter(result_dict.values()))) if result_dict else 0
        result_dict["generator_model"] = [args.model_name] * num_rows
        result_dict["generation_params"] = [json.dumps(generation_params)] * num_rows
        
        result_ds = Dataset.from_dict(result_dict)

    total_runtime = time.time() - start_time
    readme_content = f"""# Auto-Generated FastDetector Dataset
- Model Name: {args.model_name}
- Temperature: {args.temperature}
- Top P: {args.top_p}
- Top K: {args.top_k}
- Presence Penalty: {args.presence_penalty}
- Disable Thinking: {args.disable_thinking}

- Top A: {args.top_a}
- XTC Probability: {args.xtc}
- NSigma: {args.nsigma}

- Max Model Length: {args.max_model_len}
- Max Dataset Length: {args.max_dataset_len}

- Prompt File: {args.prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {args.source_dataset}
- Source Column: {args.source_column}
- Total Tokens In Processed Dataset: {tokens_processed}
- Target Num Samples: {args.num_samples}

- Total Input Tokens Processed: {total_prompt_tokens}
- Total Output Tokens Processed: {total_completion_tokens}

- Target Dataset: {args.target_dataset}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: aphrodite
"""
    upload_dataset(dataset=result_ds, dataset_name=args.target_dataset, save_locally_instead=args.save_locally_instead, cache_dir=args.cache_dir)
    if not args.save_locally_instead:
        upload_readme(dataset_name=args.target_dataset, readme_content=readme_content)
    print("Done!")

if __name__ == "__main__":
    main()
