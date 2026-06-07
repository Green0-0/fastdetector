import sys
import torch
from sentence_transformers import CrossEncoder

model_name = "Qwen/Qwen3-Reranker-4B"
print(f"Loading reranker model: {model_name}")

kwargs = {}
if "qwen3" in model_name.lower():
    kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}
    kwargs["processor_kwargs"] = {"padding_side": "left"}

try:
    model = CrossEncoder(model_name, trust_remote_code=True, **kwargs)
    print("Model loaded successfully!")
    
    pairs = [
        ("Hello world, this is a query.", "This is a matching document."),
        ("Hello world, this is a query.", "This is completely irrelevant.")
    ]
    print("Running inference...")
    scores = model.predict(pairs, batch_size=2)
    print("Inference completed successfully!")
    print("Scores:", scores)
except Exception as e:
    print(f"Error occurred: {e}", file=sys.stderr)
    sys.exit(1)
