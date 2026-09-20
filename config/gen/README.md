# Generation datasets and sampling

The folder is the dataset: `train/shard_N.toml` writes to `${gen_dataset}-train`
and `${stat_dataset}-train`, with the same rule for `val` and `test`. `N` selects the
corresponding shard from the shared `filtered` dataset. There is no Hugging
Face split setting in these configs. Thinking is disabled wherever the model
supports it; the two remaining Gemini models use their lowest supported thinking
level. Hosted GPT, Claude, and Gemini models use their provider's Batch API and do
not set sampling overrides. OpenRouter models use its standard OpenAI-compatible
endpoint, also without sampling overrides. For OpenRouter,
`disable_thinking = true` sends `reasoning = { enabled = false }` rather than a
provider-specific effort value. Inherently non-reasoning GPT models do not send
the reasoning-only `reasoning_effort` parameter.

Prompt offsets reserve each shard's original target size and are allocated in
test, validation, then training order. Test and validation reserve 2,000 prompts
per shard. Training reserves 30,000 prompts for each local-model shard and 5,000
for each hosted/API shard. Training API configs 7–10 also set
`num_samples = 5000`, which caps generation after 5,000 accepted source rows even
though their source shards contain roughly 30,000 rows. Other configs omit the
field and consume their complete source shard. The stored offsets are cumulative
rather than reduced modulo the prompt count: the schedule runs from 0 through
384,000 assignments, while `combined_dataset.json` contains 150,000 prompts.
`PromptSet` wraps those offsets modulo 150,000 at runtime, so the resulting
overlap is intentional and the unreduced values make the allocation boundaries
visible.

Gemini uses the Gemini Developer API's asynchronous Batch API as well. A minimal
pipeline block is:

```toml
[pipeline]
engine = "gemini"
model_name = "gemini-3.8-flash"
batch = true
api_key_env = "GEMINI_API_KEY"
disable_thinking = true
thinking_level = "low"
batch_state_dir = ".batch_state"
batch_poll_interval_secs = 300
max_output_tokens = 16000
```

Export the named API-key variable before running `scripts/gen.py`. Gemini batch
requests use keyed JSONL files so results remain aligned with their source rows;
`api_url` is not used because the Google SDK selects the Gemini endpoint. As
with the other hosted providers, sampling overrides are intentionally ignored
so the model's standard sampler is used. The listed Gemini 3 models do not allow
thinking to be disabled, so `thinking_level = "low"` requests their minimum.

| Shard | Dataset | Model | Sampling / transport | Source |
|---:|:---:|---|---|---|
| 0 | train | `ibm-granite/granite-4.2-30b-nvfp4` | `temperature=1.0`, `top_p=0.95` | [IBM base-model card](https://huggingface.co/ibm-granite/granite-4.2-30b#inference) |
| 1 | train | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | `temperature=0.6`, `top_p=0.95`, `top_k=20` | [Ornith model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4#quickstart) |
| 2 | train | `nvidia/Llama-3.3-70B-Instruct-NVFP4` | `temperature=0.6`, `top_p=0.9` | [checkpoint generation config](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4/blob/main/generation_config.json) |
| 3 | train | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | `temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=1.5` | [Qwen non-thinking recommendations](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) |
| 4 | train | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | `temperature=1.0`, `top_p=0.95`, `top_k=64` | [quantization model card](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit#best-practices) |
| 5 | train | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | `temperature=0.7` | [Mistral recommended settings](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4#recommended-settings) |
| 6 | train | `poolside/Laguna-S-2.1-NVFP4` | `temperature=1.0`, `top_p=1.0`, `top_k=20` | [authoritative generation config](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/blob/main/generation_config.json) |
| 7 | train | `deepseek/deepseek-v4.1-flash` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/deepseek/deepseek-v4.1-flash) |
| 8 | train | `gpt-5.6-luna` | OpenAI Batch API defaults | — |
| 9 | train | `gpt-4.1-mini` | OpenAI Batch API defaults | — |
| 10 | train | `claude-haiku-4-5-20251001` | Anthropic Batch API defaults | — |
| 11 | train | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | adjusted Aphrodite stack: `temperature=1.25`, `top_p=1.0`, `top_k=-1`, `top_a=0.1`, `xtc_probability=0.3`, `nsigma=1.5` | Requested adjusted-sampler variant |
| 12 | train | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | adjusted Aphrodite stack | Requested adjusted-sampler variant |
| 13 | train | `nvidia/Llama-3.3-70B-Instruct-NVFP4` | adjusted Aphrodite stack | Requested adjusted-sampler variant |
| 14 | train | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | adjusted Aphrodite stack | Requested adjusted-sampler variant |
| 0 | val | `TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ` | `temperature=0.7`, `top_p=0.95`, `top_k=40`, `repetition_penalty=1.1` | [AWQ model card example](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ#inference-from-python-code-using-transformers) |
| 1 | val | `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` | `temperature=0.6`, `top_p=0.9` | [checkpoint generation config](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4/blob/main/generation_config.json) |
| 2 | val | `tencent/hy3` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/tencent/hy3) |
| 3 | val | `claude-sonnet-5` | Anthropic Batch API defaults | — |
| 4 | val | `claude-sonnet-4-5-20250929` | Anthropic Batch API defaults | — |
| 5 | val | `gpt-4o` | OpenAI Batch API defaults | — |
| 6 | val | `gpt-3.5-turbo` | OpenAI Batch API defaults; 8,000-word input filter and 4,096-token output cap | [OpenAI model limits](https://developers.openai.com/api/docs/models/gpt-3.5-turbo) |
| 0 | test | `gpt-5.6-sol` | OpenAI Batch API defaults | — |
| 1 | test | `gpt-5.4` | OpenAI Batch API defaults | — |
| 2 | test | `claude-opus-5` | Anthropic Batch API defaults | — |
| 3 | test | `claude-opus-4-5-20251101` | Anthropic Batch API defaults | — |
| 4 | test | `gemini-3.8-flash` | Gemini Batch API defaults; `thinking_level=low` | — |
| 5 | test | `gemini-3.1-pro-preview` | Gemini Batch API defaults; `thinking_level=low` | — |
| 6 | test | `moonshotai/kimi-k3` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/moonshotai/kimi-k3) |
| 7 | test | `minimax/minimax-m3` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/minimax/minimax-m3) |
| 8 | test | `qwen/qwen3.7-max` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/qwen/qwen3.7-max) |
| 9 | test | `x-ai/grok-4.3` | OpenRouter defaults | [OpenRouter model](https://openrouter.ai/x-ai/grok-4.3) |

The generation jobs default to the training folder: `gen.sbatch` covers local
shards `0-6,11-14`, and `gen_api.sbatch` covers hosted shards `7-10`. Submit
validation with `sbatch --array=0-1 --export=ALL,DATASET_KIND=val slurm/gen.sbatch`
and `sbatch --array=2-6 --export=ALL,DATASET_KIND=val slurm/gen_api.sbatch`.
The test folder is API-only: `sbatch --array=0-9
--export=ALL,DATASET_KIND=test slurm/gen_api.sbatch`. For each stats job, use
the default `0-14` training array, override it with `0-6` for validation, or
with `0-9` for test.
