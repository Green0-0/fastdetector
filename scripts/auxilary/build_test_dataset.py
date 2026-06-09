import argparse
import time
from datasets import load_dataset, Dataset

def main():
    parser = argparse.ArgumentParser(description="Build a test dataset pairing Argilla text with filtered responses.")
    parser.add_argument("--argilla-dataset", type=str, default="argilla/distilabel-capybara-dpo-7k-binarized")
    parser.add_argument("--filtered-dataset", type=str, default="G-reen/cc-2021-filtered")
    parser.add_argument("--target-dataset", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    print(f"Loading {args.filtered_dataset}...")
    filtered_ds = load_dataset(args.filtered_dataset, split="train")

    print(f"Streaming {args.argilla_dataset}...")
    argilla_ds = load_dataset(args.argilla_dataset, split="train", streaming=True)

    argilla_texts = []
    for row in argilla_ds:
        if "original_response" in row and isinstance(row["original_response"], str):
            text = row["original_response"]
            if len(text) > 903:
                argilla_texts.append(text)
                if len(argilla_texts) >= args.num_samples or len(argilla_texts) >= len(filtered_ds):
                    break

    num_to_pair = min(len(argilla_texts), len(filtered_ds))
    if num_to_pair == 0:
        print("No paired data generated.")
        return

    argilla_texts = argilla_texts[:num_to_pair]
    
    # We take the original text from the filtered dataset
    filtered_originals = filtered_ds["original"][:num_to_pair]

    new_ds = Dataset.from_dict({
        "original": filtered_originals,
        "final_response": argilla_texts
    })

    print(f"Created new dataset with {len(new_ds)} rows.")
    
    if "/" in args.target_dataset:
        print(f"Pushing to hub: {args.target_dataset}...")
        new_ds.push_to_hub(args.target_dataset)
    else:
        print(f"Saving to disk: {args.target_dataset}...")
        new_ds.save_to_disk(args.target_dataset)
    print("Done!")

if __name__ == "__main__":
    main()
