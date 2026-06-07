import sys
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model_name = "Qwen/Qwen3-Embedding-4B"
print(f"Loading embedding model: {model_name}")

kwargs = {}
if "qwen3" in model_name.lower():
    kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
    kwargs["processor_kwargs"] = {"padding_side": "left"}

try:
    model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs)
    print("Model loaded successfully!")
    
    print("Loading dataset G-reen/cc-2021-rewritten...")
    ds = load_dataset("G-reen/cc-2021-rewritten", split="train")
    
    # Sort texts by length to ensure we get the longest sequences to reproduce OOM
    texts = ds["original"]
    texts = sorted(texts, key=lambda x: len(x.split()), reverse=True)
    batch_texts = texts[:8]
    
    lengths = [len(t.split()) for t in batch_texts]
    print(f"Running inference on texts with word counts: {lengths}")
    
    embeddings = model.encode(batch_texts, batch_size=4, convert_to_numpy=True, normalize_embeddings=True)
    print("Inference completed successfully!")
    print("Embeddings shape:", embeddings.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
