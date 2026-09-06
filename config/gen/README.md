# Generation datasets and sampling

The folder is the dataset: `train/shard_N.toml` writes to `${gen_dataset}-train`
and `${stat_dataset}-train`, with the same rule for `val` and `test`. `N` selects the
corresponding shard from the shared `filtered` dataset. There is no Hugging
Face split setting in these configs. Thinking is disabled in every config.
Hosted GPT and Claude models use their provider's Batch API and do not set
sampling overrides.

| Shard | Dataset | Model | Local sampling | Source |
|---:|:---:|---|---|---|
| 0 | train | `ibm-granite/granite-4.2-30b-nvfp4` | `temperature=1.0`, `top_p=0.95` | [IBM base-model card](https://huggingface.co/ibm-granite/granite-4.2-30b#inference) |
| 1 | train | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | `temperature=0.6`, `top_p=0.95`, `top_k=20` | [Ornith model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4#quickstart) |
| 2 | train | `nvidia/Llama-3.3-70B-Instruct-NVFP4` | `temperature=0.6`, `top_p=0.9` | [checkpoint generation config](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4/blob/main/generation_config.json) |
| 3 | train | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | `temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=1.5` | [Qwen non-thinking recommendations](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) |
| 4 | train | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | `temperature=0.7` | [Mistral recommended settings](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4#recommended-settings) |
| 5 | train | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | `temperature=1.0`, `top_p=0.95`, `top_k=64` | [quantization model card](https://huggingface.co/cyankiwi/gemma-4-31B-it-AWQ-4bit#best-practices) |
| 6 | train | `poolside/Laguna-S-2.1-NVFP4` | `temperature=1.0`, `top_p=1.0`, `top_k=20` | [authoritative generation config](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/blob/main/generation_config.json) |
| 7 | train | `deepseek-ai/DeepSeek-V4-Flash-0731` | `temperature=1.0`, `top_p=1.0` for non-agentic generation | [DeepSeek deployment recommendation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-locally) |
| 8 | train | `gpt-5.4-mini` | Batch API defaults | — |
| 9 | train | `claude-haiku-4-5-20251001` | Batch API defaults | — |
| 10 | train | `cyankiwi/Qwen3.8-27B-AWQ-INT4` | legacy Aphrodite stack: `temperature=1.25`, `top_p=1.0`, `top_k=-1`, `top_a=0.1`, `xtc_probability=0.3`, `nsigma=1.5` | Requested special-sampler variant |
| 11 | train | `cyankiwi/gemma-4-31B-it-AWQ-4bit` | legacy Aphrodite stack | Requested special-sampler variant |
| 12 | train | `nvidia/Llama-3.3-70B-Instruct-NVFP4` | legacy Aphrodite stack | Requested special-sampler variant |
| 13 | train | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | legacy Aphrodite stack | Requested special-sampler variant |
| 0 | val | `TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ` | `temperature=0.7`, `top_p=0.95`, `top_k=40`, `repetition_penalty=1.1` | [AWQ model card example](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ#inference-from-python-code-using-transformers) |
| 1 | val | `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` | `temperature=0.6`, `top_p=0.9` | [checkpoint generation config](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4/blob/main/generation_config.json) |
| 2 | val | `RedHatAI/Hy3-NVFP4-FP8` | `temperature=0.9`, `top_p=1.0`, `top_k=-1` | [checkpoint generation config](https://huggingface.co/RedHatAI/Hy3-NVFP4-FP8/blob/main/generation_config.json) and [Hy3 recommendations](https://huggingface.co/ApertureQA/Hy3#quickstart) |
| 3 | val | `claude-sonnet-5` | Batch API defaults | — |
| 4 | val | `gpt-5.6-luna` | Batch API defaults | — |
| 5 | val | `nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4` | legacy Aphrodite stack | Requested special-sampler variant |
| 0 | test | `gpt-5.6-sol` | Batch API defaults | — |
| 1 | test | `claude-opus-5` | Batch API defaults | — |

Hy3 uses its documented `reasoning_effort="no_think"` direct-response mode in
addition to the pipeline-wide `enable_thinking=false` template flag.

DeepSeek V4 Flash is a 304B checkpoint. Its documented vLLM deployment uses
four GPUs and checkpoint-specific server flags, so shard 7 is submitted through
`slurm/gen_large.sbatch`; the other local shards use `slurm/gen.sbatch`.

The generation job defaults cover the training folder. Submit validation with
`sbatch --array=0-2,5 --export=ALL,DATASET_KIND=val slurm/gen.sbatch` and
`sbatch --array=3-4 --export=ALL,DATASET_KIND=val slurm/gen_api.sbatch`.
The test folder is API-only: `sbatch --array=0-1
--export=ALL,DATASET_KIND=test slurm/gen_api.sbatch`. For each stats job, use
the default `0-13` training array, override it with `0-5` for validation, or
with `0-1` for test.
