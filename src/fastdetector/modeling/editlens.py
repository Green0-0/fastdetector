import glob
import os
import numpy as np
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig
from datasets import Dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.special import softmax
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from fastdetector.modeling.preprocessing import clean_text


class NormedLinear(torch.nn.Module):
    """Linear layer with preceding LayerNorm for QLoRA classification heads."""

    def __init__(self, hidden_size: int, num_labels: int, device=None, dtype=None) -> None:
        """Initialize NormedLinear.

        Args:
            hidden_size: Hidden dimension size.
            num_labels: Number of output labels.
            device: PyTorch device.
            dtype: PyTorch data type.
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm followed by Linear projection.

        Args:
            x: Input tensor of shape (batch_size, hidden_size).

        Returns:
            Output logits of shape (batch_size, num_labels).
        """
        return self.linear(self.norm(x))





def is_qlora_checkpoint(checkpoint: str) -> bool:
    """Check if a checkpoint is a QLoRA adapter.

    Args:
        checkpoint: Local directory path or HuggingFace repository ID.

    Returns:
        True if the checkpoint is a QLoRA adapter, False otherwise.
    """
    if os.path.isdir(checkpoint):
        return os.path.exists(os.path.join(checkpoint, "adapter_config.json"))
    try:
        hf_hub_download(checkpoint, "adapter_config.json")
        return True
    except Exception:
        return False


def infer_n_buckets(checkpoint: str) -> int:
    """Infer the number of classification buckets from a checkpoint.

    Args:
        checkpoint: Local directory path or HuggingFace repository ID.

    Returns:
        Number of classification buckets.

    Raises:
        ValueError: If bucket count cannot be determined.
    """
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
    """Load EditLens model and matching tokenizer.

    Args:
        checkpoint_path: Checkpoint path or HuggingFace repository ID.
        base_model_name: Base model path or HuggingFace repository ID.
        n_buckets: Number of classification buckets.

    Returns:
        Tuple of (model, tokenizer, is_qlora).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    is_qlora = is_qlora_checkpoint(checkpoint_path)

    if is_qlora:
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


def compute_editlens_scores(
    texts: list,
    model,
    tokenizer,
    is_qlora: bool,
    n_buckets: int,
    max_length: int,
    batch_size: int,
):
    """Run EditLens batch inference on input texts.

    Args:
        texts: List of input strings.
        model: Loaded EditLens model.
        tokenizer: Loaded tokenizer.
        is_qlora: Whether model is a QLoRA adapter.
        n_buckets: Number of classification buckets.
        max_length: Maximum tokenization length.
        batch_size: Inference batch size.

    Returns:
        Tuple of (bucket_preds, score_preds).
    """
    ds = Dataset.from_dict({"text": texts})

    def tokenize(example: dict) -> dict:
        """Tokenize a batch of text examples."""
        cleaned_texts = [clean_text(t) for t in example["text"]]
        return tokenizer(cleaned_texts, truncation=True, max_length=max_length)

    ds_tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    effective_batch_size = batch_size if not is_qlora else 4

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(
        ds_tokenized,
        batch_size=effective_batch_size,
        collate_fn=data_collator,
    )

    device = next(model.parameters()).device
    all_logits = []

    use_autocast = device.type == "cuda"
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            all_logits.append(outputs.logits.float().detach().cpu().numpy())

    if not all_logits:
        return [], []

    logits = np.concatenate(all_logits, axis=0)
    probs = softmax(logits, axis=1)
    bucket_preds = np.argmax(probs, axis=1)

    bucket_labels = np.arange(n_buckets)
    if n_buckets > 1:
        score_preds = (probs @ bucket_labels) / (n_buckets - 1)
    else:
        score_preds = np.zeros(len(texts), dtype=float)

    return bucket_preds.tolist(), score_preds.tolist()
