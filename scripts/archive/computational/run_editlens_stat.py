import argparse
import glob
import os
import re
import time

import emoji
import numpy as np
import torch
from datasets import Dataset
from fastdetector.utils import load_dataset_local_fallback as load_dataset
from huggingface_hub import hf_hub_download
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

from fastdetector.utils import upload_dataset, upload_readme

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
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False

def infer_n_buckets(checkpoint: str) -> int:
    """Infer the number of classification buckets from a checkpoint."""
    if not is_qlora_checkpoint(checkpoint):
        return AutoConfig.from_pretrained(checkpoint).num_labels

    if os.path.isdir(checkpoint):
        safetensor_files = glob.glob(os.path.join(checkpoint, "*.safetensors"))
        safetensor_path = safetensor_files[0] if safetensor_files else None
        bin_path = os.path.join(checkpoint, "adapter_model.bin")
    else:
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
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run EditLens inference on the dataset produced by calculate_statistics")
    parser.add_argument("--source-dataset", type=str, default="G-reen/cc-2021-rewritten-stat", help="HuggingFace dataset name to process")
    parser.add_argument("--target-dataset", type=str, default="G-reen/cc-2021-rewritten-stat-editlens", help="HuggingFace dataset name to push to")
    parser.add_argument("--checkpoint", type=str, default="pangram/editlens_roberta-large", help="Model checkpoint path or HF repo")
    parser.add_argument("--base-model", type=str, default="FacebookAI/roberta-large", help="Base model name")
    parser.add_argument("--max-length", type=int, default=512, help="Max length for tokenizer")
    parser.add_argument("--batch-size", type=int, default=24, help="Eval batch size")
    parser.add_argument("--fastdetector-prompt-metadata-column", type=str, default=None, help="Column name containing prompt metadata")
    parser.add_argument("--save-locally-instead", action="store_true", help="Save dataset locally in cached_ds folder instead of uploading")
    parser.add_argument("--cache-dir", type=str, default="cached_ds", help="Cache directory for local datasets")
    
    args = parser.parse_args()

    print(f"Downloading dataset {args.source_dataset}...")
    result_ds = load_dataset(args.source_dataset, split="train", cache_dir=args.cache_dir)

    if "original" not in result_ds.column_names or "final_response" not in result_ds.column_names:
        raise ValueError("Dataset does not appear to have 'original' and 'final_response' columns. Are you sure it was produced by calculate_statistics?")

    human_texts = result_ds["original"]
    ai_texts = result_ds["final_response"]

    print(f"Loading EditLens model from checkpoint: {args.checkpoint}")
    n_buckets = infer_n_buckets(args.checkpoint)
    print(f"Inferred n_buckets={n_buckets}")
    
    model, tokenizer, is_qlora = get_model_and_tokenizer(args.checkpoint, args.base_model, n_buckets)

    print("Computing EditLens scores for Human texts...")
    human_buckets, human_scores = compute_editlens_scores(human_texts, model, tokenizer, is_qlora, n_buckets, args.max_length, args.batch_size)
    
    print("Computing EditLens scores for AI texts...")
    ai_buckets, ai_scores = compute_editlens_scores(ai_texts, model, tokenizer, is_qlora, n_buckets, args.max_length, args.batch_size)

    for col in ["human_editlens_bucket", "human_editlens_score", "ai_editlens_bucket", "ai_editlens_score"]:
        if col in result_ds.column_names:
            result_ds = result_ds.remove_columns(col)

    result_ds = result_ds.add_column("human_editlens_bucket", human_buckets)
    result_ds = result_ds.add_column("human_editlens_score", human_scores)
    result_ds = result_ds.add_column("ai_editlens_bucket", ai_buckets)
    result_ds = result_ds.add_column("ai_editlens_score", ai_scores)

    if "editlens_model" in result_ds.column_names:
        result_ds = result_ds.remove_columns("editlens_model")
    result_ds = result_ds.add_column("editlens_model", [args.checkpoint] * len(result_ds))

    print("\nInference complete.")
    total_runtime = time.time() - start_time
    readme_content = f"""# FastDetector EditLens Statistics
- Source Dataset: {args.source_dataset}
- Target Dataset: {args.target_dataset}
- Checkpoint: {args.checkpoint}
- Base Model: {args.base_model}
- Max Length: {args.max_length}
- Batch Size: {args.batch_size}
- Inferred Buckets: {n_buckets}
- Is QLoRA: {is_qlora}
- Total Runtime: {total_runtime:.2f} seconds
"""
    upload_dataset(
        dataset=result_ds,
        dataset_name=args.target_dataset,
        save_locally_instead=args.save_locally_instead,
        cache_dir=args.cache_dir
    )
    if not args.save_locally_instead:
        upload_readme(
            dataset_name=args.target_dataset,
            readme_content=readme_content
        )
    print("Done!")

if __name__ == "__main__":
    main()
