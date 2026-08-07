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
7. Shard your raw dataset, which is where you choose how much of it to use: ``uv run scripts/shard_dataset.py --num-shards 8 --num-samples 5000000``
8. ``uv run scripts/filter.py --batch-id 0`` (after which you should run ``scripts/gen.py --gen-config config/gen/shard_0.toml``, then the ``scripts/stats/`` scripts, and finally ``scripts/analysis.py``). Every stage before the analysis takes a ``--batch-id`` naming the shard it processes.

On a Slurm cluster, ``slurm/`` has one job script per stage; the sharded stages are array jobs
(``sbatch slurm/filter.sbatch``, then ``gen``, ``stats/``, and finally ``analysis``).

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
- Several shards pushing at once causes the readme to become stale and misrepresent the dataset

### Features
- vLLM-accelerated inference for classifiers 
- Batched OAI/Anthropic/Gemini API
- New train/test data

### Code Quality
1. Cleanup the data visualization / training path
2. Merge filter.py and gen.py into one script, split off filtering behavior, update guides/readme