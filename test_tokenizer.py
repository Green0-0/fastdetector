from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
prefix = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": ""}]

full_str = tokenizer.apply_chat_template(messages, tokenize=False)
prefix_str = tokenizer.apply_chat_template(prefix, tokenize=False)

print("Full:", repr(full_str))
print("Prefix:", repr(prefix_str))

full_tokens = tokenizer.encode(full_str, add_special_tokens=False)
prefix_tokens = tokenizer.encode(prefix_str, add_special_tokens=False)

print("Full tokens:", len(full_tokens))
print("Prefix tokens:", len(prefix_tokens))
