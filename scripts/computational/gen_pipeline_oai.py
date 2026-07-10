import argparse
import os
import time
import json
from datasets import Dataset
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from fastdetector.prompting.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.utils import upload_dataset

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate LLM data using an arbitrary OpenAI-compatible endpoint.")
    parser.add_argument("--api-url", type=str, required=True, help="OpenAI-compatible API URL.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to launch.")

    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to the chat template.")

    parser.add_argument("--max-dataset-words", type=int, required=True, help="Max acceptable input length from the dataset in words.")

    parser.add_argument("--prompt-file", type=str, required=True, help="Prompt JSON file.")

    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--source-column", type=str, required=True, help="Source column.")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to stream.")
    
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    args = parser.parse_args()

    generation_params = {
        "disable_thinking": args.disable_thinking,
        "is_api_model": True,
    }

    print(f"Loading filtering prompt from file: {os.path.basename(args.prompt_file)}")
    prompt_list = load_prompts([args.prompt_file])
    prompts = PromptSet(prompt_list)

    print(f"Streaming {args.num_samples} samples from {args.source_dataset} (max {args.max_dataset_words} words)...")
    def get_len(text): return len(text.split())

    ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)
    samples = []
    words_processed = 0
    for row in ds:
        text = row[args.source_column]
        text = str(text) if text is not None else ""
        num_words = get_len(text)
        if num_words <= args.max_dataset_words:
            samples.append(text)
            words_processed += num_words
        if len(samples) >= args.num_samples:
            break
    print(f"Loaded {len(samples)} samples.")

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    
    print(f"Using API endpoint: {args.api_url}")
    result_dict, total_prompt_tokens, total_completion_tokens = build_dataset(
        samples=samples,
        api_url=args.api_url,
        prompts=prompts,
        generation_params=generation_params,
        api_key=api_key,
        model_name=args.model_name,
    )
    
    num_rows = len(next(iter(result_dict.values()))) if result_dict else 0
    result_dict["generator_model"] = [args.model_name] * num_rows
    result_dict["generation_params"] = [json.dumps(generation_params)] * num_rows
    
    result_ds = Dataset.from_dict(result_dict)

    total_runtime = time.time() - start_time
    readme_content = f"""# Auto-Generated FastDetector Dataset
- Model Name: {args.model_name}
- Disable Thinking: {args.disable_thinking}

- Max Dataset Words: {args.max_dataset_words}

- Prompt File: {args.prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {args.source_dataset}
- Source Column: {args.source_column}
- Total Words In Processed Dataset: {words_processed}
- Target Num Samples: {args.num_samples}

- Total Input Tokens Processed: {total_prompt_tokens}
- Total Output Tokens Processed: {total_completion_tokens}

- Target Dataset: {args.target_dataset}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: OpenAI-Compatible
"""
    upload_dataset(dataset=result_ds, dataset_name=args.target_dataset, readme_content=readme_content, save_locally_instead=args.save_locally_instead, cache_dir=args.cache_dir)
    print("Done!")

if __name__ == "__main__":
    main()
