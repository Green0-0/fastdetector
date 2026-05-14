import glob
import os
from datasets import Dataset, load_dataset
from fastdetector.prompts import load_prompts
from fastdetector.generator import build_dataset

# --- Configuration ---
SOURCE_DATASET = "G-reen/view"
SOURCE_CONFIG = None
SOURCE_COLUMN = "trafilatura_text"
NUM_SAMPLES = 1_000

TARGET_DATASET = "G-reen/cc-contiguous"
FILTERED_COLUMN = "response_0"
ORIGINAL_COLUMN = "original"

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "sample_prompts", "filtering")


def add_validation_columns(dataset: Dataset, original_col: str, filtered_col: str) -> Dataset:
    """Append validation columns for contiguous subset checks."""

    def _check_batch(batch: dict) -> dict:
        originals = batch[original_col]
        filtered = batch[filtered_col]
        malformed: list[bool] = []
        deviation_size: list[int] = []
        for original, filtered_text in zip(originals, filtered):
            if not isinstance(original, str):
                original = "" if original is None else str(original)
            if not isinstance(filtered_text, str):
                filtered_text = "" if filtered_text is None else str(filtered_text)
            malformed.append(not filtered_text or filtered_text not in original)
            original_words = len(original.split())
            filtered_words = len(filtered_text.split())
            deviation_size.append(max(0, original_words - filtered_words))
        return {"malformed": malformed, "deviation_size": deviation_size}

    return dataset.map(_check_batch, batched=True)


def load_samples(dataset: str, config: str | None, column: str, num_samples: int) -> list[str]:
    """Stream a HuggingFace dataset and extract the first num_samples texts."""
    print(f"Streaming {num_samples} samples from {dataset} ({config})...")
    if config:
        ds = load_dataset(dataset, name=config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset, split="train", streaming=True)
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

    # Filter and push
    build_dataset(
        samples=samples,
        target=TARGET_DATASET,
        api_url=api_url,
        prompts=prompts,
        append=False,
    )

    # Validate that filtered outputs are contiguous subsets of the originals
    print("Validating filtered outputs...")
    result_ds = load_dataset(TARGET_DATASET, split="train")
    result_ds = add_validation_columns(result_ds, original_col=ORIGINAL_COLUMN, filtered_col=FILTERED_COLUMN)
    malformed_count = sum(result_ds["malformed"])
    print(f"Malformed rows: {malformed_count}/{len(result_ds)}")
    result_ds.push_to_hub(TARGET_DATASET)
    print("Done!")


if __name__ == "__main__":
    main()
