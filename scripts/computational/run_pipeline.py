from transformers import AutoTokenizer
import argparse
import os
import time
from datasets import load_dataset, Dataset
from fastdetector.prompts import PromptSet, load_prompts
from fastdetector.generator import build_dataset
from fastdetector.utils import upload_dataset
from fastdetector.llm_utils import llm_server_context

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate LLM data using an LLM server.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name to launch.")
    parser.add_argument("--temperature", type=float, required=True, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, required=True, help="Generation top-p.")
    parser.add_argument("--presence-penalty", type=float, required=True, help="Generation presence penalty.")

    parser.add_argument("--max-model-len", type=int, required=True, help="Max model length.")
    parser.add_argument("--max-dataset-len", type=int, required=True, help="Max acceptable input length from the dataset.")

    parser.add_argument("--prompt-file", type=str, required=True, help="Prompt JSON file.")

    parser.add_argument("--source-dataset", type=str, required=True, help="Source dataset.")
    parser.add_argument("--source-column", type=str, required=True, help="Source column.")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to stream.")
    
    parser.add_argument("--target-dataset", type=str, required=True, help="Target dataset.")
    args = parser.parse_args()

    generation_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
    }

    print(f"Loading filtering prompt from file: {os.path.basename(args.prompt_file)}")
    prompt_list = load_prompts([args.prompt_file])
    prompts = PromptSet(prompt_list)

    print(f"Loading tokenizer and streaming {args.num_samples} samples from {args.source_dataset}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ds = load_dataset(args.source_dataset, split="train", streaming=True)
    
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

    with llm_server_context(engine="vllm", model_name=args.model_name, port=None, max_model_len=args.max_model_len) as api_url:
        print(f"Using API endpoint: {api_url}")
        result_dict = build_dataset(
            samples=samples,
            api_url=api_url,
            prompts=prompts,
            generation_params=generation_params,
        )
        result_ds = Dataset.from_dict(result_dict)

    total_runtime = time.time() - start_time
    readme_content = f"""# Auto-Generated FastDetector Dataset
- Model Name: {args.model_name}
- Temperature: {args.temperature}
- Top P: {args.top_p}
- Presence Penalty: {args.presence_penalty}

- Max Model Length: {args.max_model_len}
- Max Dataset Length: {args.max_dataset_len}

- Prompt File: {args.prompt_file}
- Total Train Prompts: {len(prompts.get_train())}

- Source Dataset: {args.source_dataset}
- Source Column: {args.source_column}
- Total Tokens In Processed Dataset: {tokens_processed}
- Target Num Samples: {args.num_samples}

- Target Dataset: {args.target_dataset}

- Total Runtime: {total_runtime:.2f} seconds

- Engine: vllm
"""

    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        readme_content=readme_content
    )

    print("Done!")

if __name__ == "__main__":
    main()
