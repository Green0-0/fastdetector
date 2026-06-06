import argparse
import os
import time
from huggingface_hub import HfApi
from datasets import load_dataset
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context

GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
}

PROMPT_FILE = "prompts/combined_dataset.json"

def load_samples(dataset: str, column: str, num_samples: int) -> list[str]:
    """Stream a HuggingFace dataset and extract the first num_samples texts."""
    print(f"Streaming {num_samples} samples from {dataset}...")
    ds = load_dataset(dataset, split="train", streaming=True)
    samples = [row[column] for row in ds.take(num_samples)]
    print(f"Loaded {len(samples)} samples.")
    return samples

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate LLM data using an LLM server.")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E4B-it", help="Model name to launch.")
    parser.add_argument("--source-dataset", type=str, default="G-reen/cc-2021-filtered", help="Source dataset.")
    parser.add_argument("--source-column", type=str, default="response_0", help="Source column.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to stream.")
    parser.add_argument("--target-dataset", type=str, default="G-reen/cc-2021-rewritten", help="Target dataset.")
    args = parser.parse_args()

    with llm_server_context(engine="vllm", model_name=args.model_name, port=None) as api_url:
        print(f"Using API endpoint: {api_url}")

        if not os.path.exists(PROMPT_FILE):
            raise FileNotFoundError(f"Prompt JSON file not found: {PROMPT_FILE}")

        print(f"Loading prompts from file: {os.path.basename(PROMPT_FILE)}")
        prompt_list = load_prompts([PROMPT_FILE])
        prompts = PromptSet(prompt_list)
        print(f"Total prompts loaded: {len(prompts.get_train())}")

        # Stream the source dataset
        samples = load_samples(args.source_dataset, args.source_column, args.num_samples)

        # Generate locally
        result_ds = build_dataset(
            samples=samples,
            target=args.target_dataset,
            api_url=api_url,
            prompts=prompts,
            append=False,
            generation_params=GENERATION_PARAMS,
        )

    result_ds.push_to_hub(args.target_dataset)
    print(f"Dataset pushed to '{args.target_dataset}' with {len(result_ds)} rows and {len(result_ds.column_names)} columns.")
    
    total_runtime = time.time() - start_time
    
    readme_content = f"""# Generation Configuration
- Source Dataset: {args.source_dataset}
- Source Column: {args.source_column}
- Num Samples: {args.num_samples}
- Generation Params: {GENERATION_PARAMS}
- Model Name: {args.model_name}
- Engine: vllm
- Total Runtime: {total_runtime:.2f} seconds
"""
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.target_dataset,
            repo_type="dataset"
        )
        print("Configuration details written to Dataset README.md on HuggingFace Hub.")
    except Exception as e:
        print(f"Error uploading README to HuggingFace Hub: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
