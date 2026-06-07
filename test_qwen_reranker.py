import sys
import torch
from datasets import load_dataset
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
    
    print("Loading dataset G-reen/cc-2021-rewritten...")
    ds = load_dataset("G-reen/cc-2021-rewritten", split="train")
    
    # Sort texts by length to test the longest sequences
    texts = ds["original"]
    texts = sorted(texts, key=lambda x: len(x.split()), reverse=True)
    
    # Make pairs of the longest texts
    pairs = [(texts[i], texts[i+1]) for i in range(0, 4, 2)]
    
    lengths = [(len(p[0].split()), len(p[1].split())) for p in pairs]
    print(f"Running inference on pairs with word counts: {lengths}")
    
    scores = model.predict(pairs, batch_size=2)
    print("Inference completed successfully!")
    print("Scores:", scores)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
