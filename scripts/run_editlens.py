import argparse
import glob
import os
import re

import emoji
import numpy as np
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

class NormedLinear(torch.nn.Module):
    """Linear layer preceded by LayerNorm to keep logits well-scaled."""

    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))
    
def clean_text(text):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = emoji.demojize(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

def count_words(text):
    if text is None:
        return 0
    return len(re.findall(r"\b\w+\b", text))

def is_qlora_checkpoint(checkpoint: str) -> bool:
    """Check whether checkpoint is a QLora adapter (local path or HF repo)."""
    if os.path.isdir(checkpoint):
        return os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    # HF repo — check if adapter_config.json exists in the repo
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False


def infer_n_buckets(checkpoint: str) -> int:
    """Infer the number of classification buckets from a checkpoint."""
    if not is_qlora_checkpoint(checkpoint):
        return AutoConfig.from_pretrained(checkpoint).num_labels

    # LoRA: find n_buckets from the saved score head weight shape
    if os.path.isdir(checkpoint):
        safetensor_files = glob.glob(os.path.join(checkpoint, "*.safetensors"))
        safetensor_path = safetensor_files[0] if safetensor_files else None
        bin_path = os.path.join(checkpoint, "adapter_model.bin")
    else:
        # HF repo — download the adapter weights
        try:
            safetensor_path = hf_hub_download(checkpoint, "adapter_model.safetensors")
        except Exception:
            safetensor_path = None
        try:
            bin_path = hf_hub_download(checkpoint, "adapter_model.bin")
        except Exception:
            bin_path = None

    if safetensor_path and os.path.exists(safetensor_path):
        with safe_open(safetensor_path, framework="pt") as f:
            for key in f.keys():
                if "score" in key and "linear.weight" in key:
                    return f.get_tensor(key).shape[0]

    if bin_path and os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        for key, tensor in state_dict.items():
            if "score" in key and "linear.weight" in key:
                return tensor.shape[0]

    raise ValueError(
        f"Could not infer n_buckets from checkpoint at {checkpoint}. "
        "No score head weights found in adapter."
    )


def get_model_and_tokenizer(checkpoint_path: str, base_model_name: str, n_buckets: int):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    is_qlora = is_qlora_checkpoint(checkpoint_path)

    if is_qlora:
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=n_buckets,
            quantization_config=quantization_config,
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base_model, "score") and isinstance(base_model.score, torch.nn.Linear):
            hidden_size = base_model.config.hidden_size
            device = next(base_model.parameters()).device
            base_model.score = NormedLinear(hidden_size, n_buckets, device=device)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
        )

    model.eval()
    return model, tokenizer, is_qlora

def compute_editlens_scores(texts: list, model, tokenizer, is_qlora: bool, n_buckets: int, max_length: int, batch_size: int):
    ds = Dataset.from_dict({"text": texts})
    
    def tokenize(example):
        cleaned_texts = [clean_text(t) for t in example["text"]]
        return tokenizer(cleaned_texts, truncation=True, max_length=max_length)
        
    ds_tokenized = ds.map(tokenize, num_proc=4, batched=True, remove_columns=["text"])
    
    training_args = TrainingArguments(
        output_dir="/tmp/editlens_inference",
        per_device_eval_batch_size=batch_size if not is_qlora else 4,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    output = trainer.predict(ds_tokenized)
    probs = softmax(output.predictions, axis=1)
    bucket_preds = np.argmax(probs, axis=1)

    bucket_labels = np.arange(n_buckets)
    score_preds = (probs @ bucket_labels) / (n_buckets - 1)
    
    return bucket_preds.tolist(), score_preds.tolist()

def main():
    parser = argparse.ArgumentParser(description="Run EditLens inference on the dataset produced by calculate_statistics")
    parser.add_argument("--source-dataset", type=str, default="G-reen/cc-2021-rewritten", help="HuggingFace dataset name to process")
    parser.add_argument("--checkpoint", type=str, default="pangram/editlens_roberta-large", help="Model checkpoint path or HF repo")
    parser.add_argument("--base-model", type=str, default="FacebookAI/roberta-large", help="Base model name")
    parser.add_argument("--max-length", type=int, default=512, help="Max length for tokenizer")
    parser.add_argument("--batch-size", type=int, default=24, help="Eval batch size")
    
    args = parser.parse_args()

    print(f"Downloading dataset {args.source_dataset}...")
    result_ds = load_dataset(args.source_dataset, split="train")

    if "original" not in result_ds.column_names or "final_response_index" not in result_ds.column_names:
        raise ValueError("Dataset does not appear to have 'original' and 'final_response_index' columns. Are you sure it was produced by calculate_statistics?")

    human_texts = result_ds["original"]
    resp_cols = {col: result_ds[col] for col in result_ds.column_names if col.startswith("response_")}
    final_indices = result_ds["final_response_index"]
    ai_texts = [resp_cols[f"response_{idx}"][i] for i, idx in enumerate(final_indices)]

    print(f"Loading EditLens model from checkpoint: {args.checkpoint}")
    n_buckets = infer_n_buckets(args.checkpoint)
    print(f"Inferred n_buckets={n_buckets}")
    
    model, tokenizer, is_qlora = get_model_and_tokenizer(args.checkpoint, args.base_model, n_buckets)

    print("Computing EditLens scores for Human texts...")
    human_buckets, human_scores = compute_editlens_scores(human_texts, model, tokenizer, is_qlora, n_buckets, args.max_length, args.batch_size)
    
    print("Computing EditLens scores for AI texts...")
    ai_buckets, ai_scores = compute_editlens_scores(ai_texts, model, tokenizer, is_qlora, n_buckets, args.max_length, args.batch_size)

    # Remove existing columns if they are present to avoid ValueError on rerun
    for col in ["human_editlens_bucket", "human_editlens_score", "ai_editlens_bucket", "ai_editlens_score"]:
        if col in result_ds.column_names:
            result_ds = result_ds.remove_columns(col)

    # Prefix columns with a tag derived from base_model if wanted, but standardizing to human/ai is consistent with calculate_statistics
    result_ds = result_ds.add_column("human_editlens_bucket", human_buckets)
    result_ds = result_ds.add_column("human_editlens_score", human_scores)
    result_ds = result_ds.add_column("ai_editlens_bucket", ai_buckets)
    result_ds = result_ds.add_column("ai_editlens_score", ai_scores)

    print("\nInference complete.")
    print(f"Human score stats: mean={np.mean(human_scores):.4f}, std={np.std(human_scores):.4f}")
    print(f"AI score stats: mean={np.mean(ai_scores):.4f}, std={np.std(ai_scores):.4f}")

    ai_true = np.array(ai_scores)
    human_true = np.array(human_scores)
    
    threshold = 0.5
    tp = np.sum(ai_true > threshold)
    fn = np.sum(ai_true <= threshold)
    tn = np.sum(human_true <= threshold)
    fp = np.sum(human_true > threshold)
    
    total = len(ai_true) + len(human_true)
    accuracy = (tp + tn) / total if total > 0 else 0
    
    stats_md = f"""
## EditLens Inference Statistics
- **Checkpoint**: {args.checkpoint}
- **Base Model**: {args.base_model}

### Score Distributions
- **Human Scores**: Mean = {np.mean(human_scores):.4f}, Std = {np.std(human_scores):.4f}
- **AI Scores**: Mean = {np.mean(ai_scores):.4f}, Std = {np.std(ai_scores):.4f}

### Classification (Threshold = {threshold})
- **Overall Accuracy**: {accuracy * 100:.2f}%
- **True Positives (AI correctly identified)**: {tp} ({(tp / len(ai_true) * 100) if len(ai_true) else 0:.2f}%)
- **False Negatives (AI missed)**: {fn} ({(fn / len(ai_true) * 100) if len(ai_true) else 0:.2f}%)
- **True Negatives (Human correctly identified)**: {tn} ({(tn / len(human_true) * 100) if len(human_true) else 0:.2f}%)
- **False Positives (Human misclassified as AI)**: {fp} ({(fp / len(human_true) * 100) if len(human_true) else 0:.2f}%)
"""

    try:
        api = HfApi()
        try:
            existing_readme_path = hf_hub_download(repo_id=args.source_dataset, filename="README.md", repo_type="dataset")
            with open(existing_readme_path, "r") as f:
                existing_readme = f.read()
            final_readme = existing_readme + "\n\n" + stats_md
        except Exception:
            final_readme = stats_md

        api.upload_file(
            path_or_fileobj=final_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.source_dataset,
            repo_type="dataset"
        )
        print("Appended EditLens statistics to Dataset README.md on HuggingFace Hub.")
    except Exception as e:
        print(f"Error updating README on HuggingFace Hub: {e}")

    print(f"Pushing updated dataset back to {args.source_dataset}...")
    result_ds.push_to_hub(args.source_dataset)
    print("Done!")

if __name__ == "__main__":
    main()
