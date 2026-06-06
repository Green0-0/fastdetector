from datasets import config
import argparse
import os
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.llm_utils import llm_server_context

# --- Configuration ---
FILTERING_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.0,
}

PROMPT_FILE = "prompts/filter_contiguous_subset.json"

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

def check_batch(batch: dict) -> dict:
    originals = batch["original"]
    filtered = batch["response_0"]
    malformed: list[bool] = []
    deviation_size: list[int] = []
    updated_filtered: list[str] = []
    for original, filtered_text in zip(originals, filtered):
        original = str(original) if original is not None else ""
        filtered_text = str(filtered_text) if filtered_text is not None else ""
        
        orig_norm = original.translate(PUNCT_TRANSLATION)
        filt_norm = filtered_text.translate(PUNCT_TRANSLATION)
        
        orig_canon_parts = []
        orig_mapping = []
        for i, c in enumerate(original):
            norm_c = c.translate(PUNCT_TRANSLATION)
            for nc in norm_c:
                if not nc.isspace():
                    lowered = nc.lower()
                    orig_canon_parts.append(lowered)
                    orig_mapping.extend([i] * len(lowered))
        orig_canon = "".join(orig_canon_parts)
        
        filt_canon = "".join(c.lower() for c in filt_norm if not c.isspace())
        
        if not filt_canon or filt_canon not in orig_canon:
            malformed.append(True)
            updated_filtered.append(filtered_text)
        else:
            malformed.append(False)
            start_idx = orig_canon.find(filt_canon)
            end_idx = start_idx + len(filt_canon) - 1
            orig_start = orig_mapping[start_idx]
            orig_end = orig_mapping[end_idx]
            updated_filtered.append(original[orig_start:orig_end+1])
            
        original_words = len(orig_norm.split())
        filtered_words = len(filt_norm.split())
        deviation_size.append(max(0, original_words - filtered_words))
        
    return {"malformed": malformed, "deviation_size": deviation_size, "response_0": updated_filtered}
    
def main():
    parser = argparse.ArgumentParser(description="Filter data using an LLM server.")
    parser.add_argument("--model-name", type=str, default="google/gemma-4-E4B-it", help="Model name to launch.")
    parser.add_argument("--max-model-len", type=int, default=32000, help="Max model length.")
    parser.add_argument("--source-dataset", type=str, default="G-reen/cc-2021-raw", help="Source dataset.")
    parser.add_argument("--source-column", type=str, default="trafilatura_text", help="Source column.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to stream.")
    parser.add_argument("--target-dataset", type=str, default="G-reen/cc-2021-filtered", help="Target dataset.")
    args = parser.parse_args()

    if not os.path.exists(PROMPT_FILE):
        raise FileNotFoundError(f"Prompt JSON file not found: {PROMPT_FILE}")

    print(f"Loading filtering prompt from file: {os.path.basename(PROMPT_FILE)}")
    prompt_list = load_prompts([PROMPT_FILE])
    prompts = PromptSet(prompt_list)

    print(f"Loading tokenizer and streaming {args.num_samples} samples from {args.source_dataset}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_length = args.max_model_len // 2 - 1000
    ds = load_dataset(args.source_dataset, split="train", streaming=True)
    
    samples = []
    for row in ds:
        text = row[args.source_column]
        text = str(text) if text is not None else ""
        if len(tokenizer.encode(text)) <= max_length:
            samples.append(text)
        if len(samples) >= args.num_samples:
            break
    print(f"Loaded {len(samples)} samples.")

    with llm_server_context(engine="vllm", model_name=args.model_name, port=None, max_model_len=args.max_model_len) as api_url:
        print(f"Using API endpoint: {api_url}")

        result_ds = build_dataset(
            samples=samples,
            target=args.target_dataset,
            api_url=api_url,
            prompts=prompts,
            append=False,
            generation_params=FILTERING_GENERATION_PARAMS,
        )

    # Validate that filtered outputs are contiguous subsets of the originals
    print("Validating filtered outputs...")
    result_ds = result_ds.map(check_batch, batched=True)
    malformed_count = sum(result_ds["malformed"])
    print(f"Malformed rows: {malformed_count}/{len(result_ds)}")
    result_ds.push_to_hub(args.target_dataset)
    print("Done!")

if __name__ == "__main__":
    main()
