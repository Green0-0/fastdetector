import glob
import os
from datasets import load_dataset
from fastdetector.prompts import load_prompts
from fastdetector.generator import build_dataset

# --- Configuration ---
SOURCE_DATASET = "HuggingFaceFW/fineweb-edu"
SOURCE_CONFIG = "sample-10BT"
SOURCE_COLUMN = "text"
NUM_SAMPLES = 10_000

TARGET_DATASET = "G-reen/fineweb-edu-rewritten"  # TODO: set this

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "sample_prompts")


def load_samples(dataset: str, config: str, column: str, num_samples: int) -> list[str]:
    """Stream a HuggingFace dataset and extract the first num_samples texts."""
    print(f"Streaming {num_samples} samples from {dataset} ({config})...")
    ds = load_dataset(dataset, name=config, split="train", streaming=True)
    samples = [row[column] for row in ds.take(num_samples)]
    print(f"Loaded {len(samples)} samples.")
    return samples


def main():
    # Get the API URL from environment (set by the sbatch script)
    api_url = os.environ.get("VLLM_API_URL")
    if not api_url:
        raise RuntimeError("VLLM_API_URL environment variable is not set.")

    print(f"Using API endpoint: {api_url}")

    # Load all prompt JSON files
    prompt_files = sorted(glob.glob(os.path.join(PROMPT_DIR, "*.json")))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt JSON files found in {PROMPT_DIR}")

    print(f"Loading prompts from {len(prompt_files)} files:")
    for pf in prompt_files:
        print(f"  - {os.path.basename(pf)}")

    prompts = load_prompts(prompt_files)
    prompts.shuffle(seed=42)
    print(f"Total prompts loaded: {len(prompts.get_train())}")

    # Stream the source dataset
    samples = load_samples(SOURCE_DATASET, SOURCE_CONFIG, SOURCE_COLUMN, NUM_SAMPLES)

    # Generate and push
    build_dataset(
        samples=samples,
        target=TARGET_DATASET,
        api_url=api_url,
        prompts=prompts,
        append=False,
    )

    print("Done!")


if __name__ == "__main__":
    main()
