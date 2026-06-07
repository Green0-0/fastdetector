import sys
import torch
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
    
    texts = ["Hello world, this is a test.", "Another test text for embedding."]
    print("Running inference...")
    embeddings = model.encode(texts, batch_size=8, convert_to_numpy=True, normalize_embeddings=True)
    print("Inference completed successfully!")
    print("Embeddings shape:", embeddings.shape)
except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    sys.exit(1)
