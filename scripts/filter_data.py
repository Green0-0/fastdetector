import argparse
import os
from datasets import Dataset, load_dataset
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context

# --- Configuration ---
SOURCE_DATASET = "G-reen/view"
SOURCE_CONFIG = None
SOURCE_COLUMN = "trafilatura_text"
NUM_SAMPLES = 1_000

TARGET_DATASET = "G-reen/cc-contiguous"
FILTERED_COLUMN = "response_0"
ORIGINAL_COLUMN = "original"
FILTERING_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
}

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

PUNCT_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": "\"",
    "\u201D": "\"",
    "\u201A": "'",
    "\u201E": "\"",
    "\u2013": "-",
    "\u2014": "-",
    "\u2011": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00AB": "\"",
    "\u00BB": "\"",
    "\u00A0": " ",
    "\u2009": " ",
    "\u202F": " ",
})

def normalize_text(text: str) -> str:
    """Normalize punctuation style for subset checks."""
    return text.translate(PUNCT_TRANSLATION)

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
            original_norm = normalize_text(original)
            filtered_norm = normalize_text(filtered_text)
            malformed.append(not filtered_norm or filtered_norm not in original_norm)
            original_words = len(original_norm.split())
            filtered_words = len(filtered_norm.split())
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
    parser = argparse.ArgumentParser(description="Filter data using an LLM server.")
    parser.add_argument("--engine", type=str, default="vllm", help="LLM engine to run (e.g. vllm).")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E4B", help="Model name to launch.")
    parser.add_argument("--port", type=int, default=None, help="Port to run LLM server on (default: auto-detect free port).")
    args = parser.parse_args()

    with llm_server_context(engine=args.engine, model_name=args.model_name, port=args.port) as api_url:
        print(f"Using API endpoint: {api_url}")

        # Load prompt JSON file
        prompt_file = os.path.join(PROMPT_DIR, "filter_contiguous_subset.json")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt JSON file not found: {prompt_file}")

        print(f"Loading prompts from file: {os.path.basename(prompt_file)}")

        prompt_list = load_prompts([prompt_file])
        prompts = PromptSet(prompt_list)
        prompts.shuffle(seed=42)
        print(f"Total prompts loaded: {len(prompts.get_train())}")

        # Stream the source dataset
        samples = load_samples(SOURCE_DATASET, SOURCE_CONFIG, SOURCE_COLUMN, NUM_SAMPLES)

        # Filter locally
        result_ds = build_dataset(
            samples=samples,
            target=TARGET_DATASET,
            api_url=api_url,
            prompts=prompts,
            append=False,
            generation_params=FILTERING_GENERATION_PARAMS,
        )

        # Validate that filtered outputs are contiguous subsets of the originals
        print("Validating filtered outputs...")
        result_ds = add_validation_columns(result_ds, original_col=ORIGINAL_COLUMN, filtered_col=FILTERED_COLUMN)
        malformed_count = sum(result_ds["malformed"])
        print(f"Malformed rows: {malformed_count}/{len(result_ds)}")
        result_ds.push_to_hub(TARGET_DATASET)
        print("Done!")

if __name__ == "__main__":
    main()
