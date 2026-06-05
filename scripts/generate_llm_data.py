import argparse
import os
import time
from huggingface_hub import HfApi
from datasets import load_dataset
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context

# --- Configuration ---
SOURCE_DATASET = "G-reen/cc-contiguous"
SOURCE_COLUMN = "response_0"
NUM_SAMPLES = 100

TARGET_DATASET = "G-reen/cc-contiguous-rewritten"

GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "presence_penalty": 1.5,
}

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

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
    parser.add_argument("--engine", type=str, default="vllm", help="LLM engine to run (e.g. vllm).")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E4B-it", help="Model name to launch.")
    parser.add_argument("--port", type=int, default=None, help="Port to run LLM server on (default: auto-detect free port).")
    args = parser.parse_args()

    with llm_server_context(engine=args.engine, model_name=args.model_name, port=args.port) as api_url:
        print(f"Using API endpoint: {api_url}")

        # Load prompt JSON files
        prompt_files = [
            os.path.join(PROMPT_DIR, "combined_dataset.json"),
        ]
        for pf in prompt_files:
            if not os.path.exists(pf):
                raise FileNotFoundError(f"Prompt JSON file not found: {pf}")

        print(f"Loading prompts from {len(prompt_files)} files:")
        for pf in prompt_files:
            print(f"  - {os.path.basename(pf)}")

        prompt_list = load_prompts(prompt_files)
        prompts = PromptSet(prompt_list)
        print(f"Total prompts loaded: {len(prompts.get_train())}")

        # Stream the source dataset
        samples = load_samples(SOURCE_DATASET, SOURCE_COLUMN, NUM_SAMPLES)

        # Generate locally
        result_ds = build_dataset(
            samples=samples,
            target=TARGET_DATASET,
            api_url=api_url,
            prompts=prompts,
            append=False,
            generation_params=GENERATION_PARAMS,
        )

    result_ds.push_to_hub(TARGET_DATASET)
    print(f"Dataset pushed to '{TARGET_DATASET}' with {len(result_ds)} rows and {len(result_ds.column_names)} columns.")
    
    total_runtime = time.time() - start_time
    
    readme_content = f"""# Generation Configuration
- Source Dataset: {SOURCE_DATASET}
- Source Column: {SOURCE_COLUMN}
- Num Samples: {NUM_SAMPLES}
- Generation Params: {GENERATION_PARAMS}
- Model Name: {args.model_name}
- Engine: {args.engine}
- Total Runtime: {total_runtime:.2f} seconds
"""
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=TARGET_DATASET,
            repo_type="dataset"
        )
        print("Configuration details written to Dataset README.md on HuggingFace Hub.")
    except Exception as e:
        print(f"Error uploading README to HuggingFace Hub: {e}")

    print("Done!")

if __name__ == "__main__":
    main()
