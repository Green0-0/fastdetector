import glob
import os
import re
import emoji
import numpy as np
import torch
from datasets import Dataset
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

class NormedLinear(torch.nn.Module):
    """Linear layer preceded by LayerNorm to keep logits well-scaled."""
    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))
    
def clean_text(text):
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    text = emoji.demojize(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

def is_qlora_checkpoint(checkpoint: str) -> bool:
    if os.path.isdir(checkpoint):
        return os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False

def infer_n_buckets(checkpoint: str) -> int:
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

    raise ValueError(f"Could not infer n_buckets from checkpoint at {checkpoint}.")

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
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

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
