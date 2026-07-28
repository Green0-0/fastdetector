# FastDetector
FastDetector is a pipeline for end-to-end development of AI text detectors, with enough flexibility to also filter and generate synthetic data for any domain!

## Quick Start
1. Install uv: ``pip install uv``
2. Install FastDetector: ``uv sync``
3. Install flash attention for qwen embedding models: 

    a. Find an appropriate wheel at https://github.com/mjun0812/flash-attention-prebuild-wheels/

    b. Install it (ie: ``uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl``)
4. Create a seperate venv for vLLM (used by the generation/filtering pipelines; the statistics scripts run in the main venv): ``uv venv .vllm --python 3.12``
5. Activate it: ``source .vllm/bin/activate``
6. Install vLLM: https://docs.vllm.ai/en/latest/getting_started/installation/
    - ie: ``uv pip install vllm --torch-backend cu130``
7. ``uv run scripts/filter.py`` (after which you should run ``scripts/gen.py --gen-config config/gen/shard_0.toml``, then the ``scripts/stats/`` scripts, and finally ``scripts/analysis.py``)

## Testing
```
uv pip install "pytest>=8.0"
uv run pytest          # fast, offline, CPU-only (~20s)
uv run pytest -m gpu   # VRAM preflight against the real config/ checkpoints
```
Expensive tests (`gpu`, `slow`, `network`, `vllm`) are opt-in markers and are
deselected by default. See [tests/README.md](tests/README.md) for the tiers, the
environment variables, and how to smoke out OOMs before submitting a job.

## Documentation
*Note: Please read the general guide first.*

[General Guide](docs/general_guide.md) - Explains what to run and where to configure things!

[Dataset Guide](docs/dataset_guide.md) - Build your first human-AI paired dataset!

[Modeling Guide](docs/model_guide.md) - Train a classifier on your data!

[Special: Synthesizing textbook quality data](docs/textbook_guide.md) - Not interested in classifiers? Want to build pi-style pretraining data instead? 

[Special: A filtered creative writing dataset](docs/creative_writing.md) - How about using text detectors to filter for human-like AI writing?

## Agent Help
To all LLMs, refer to [SKILL.md](SKILL.md).

## WIP

### Issues
- OOM when calculating stats (`uv run pytest -m gpu` measures the real peak against config/)
- Have an LLM inspect the dataset outputs row by row

### Features
- Training classifiers with unsloth
- vLLM-accelerated inference for classifiers 
- Automatic hyperparameter sweeping with Optuna

### Code Quality
1. Merge filter.py and gen.py into one script, split off filtering behavior, update general_guide.md
2. Make globals.toml specify a username and dataset paths instead of prefixes, update general_guide.md
