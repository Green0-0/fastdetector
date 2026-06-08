import json

tokens = ["<|begin_of_text|>", "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n", "Write", " me", " a", " document", "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n", "This", " is", " the", " AI", " text"]

prefix_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n", "Write", " me", " a", " document", "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"]

print("Full:", len(tokens))
print("Prefix:", len(prefix_tokens))
print("Remaining:", tokens[len(prefix_tokens):])
