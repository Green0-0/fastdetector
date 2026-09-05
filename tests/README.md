# FastDetector test suite

One suite, four opt-in tiers. Expensive tiers are **deselected by default**, so
a bare `pytest` is always safe to run on a laptop and never touches the network.

```bash
uv pip install "pytest>=8.0"     # or: uv sync --extra test
uv run pytest                    # the default tier: ~800 tests, ~20s, offline
```

Run it from the repository root — pytest only applies the configured `testpaths`
when invoked from the rootdir. From elsewhere, pass the path: `pytest /path/to/tests`.

## Tiers

Tiers are markers, declared in `pyproject.toml`. The default `addopts` line
deselects the expensive ones; a `-m` on the command line **replaces** that
filter, so selecting a tier is a single flag.

| Marker | Meaning | How to run |
|---|---|---|
| *(none)* | Fast, offline, CPU-only. Runs on every commit. | `pytest` |
| `slow` | More than a few seconds. | `pytest -m slow` |
| `network` | Downloads from the HF Hub. | `pytest -m network` |
| `gpu` | Needs a CUDA device. | `pytest -m gpu` |
| `bigmem` | Needs a lot of host RAM. | `pytest -m "gpu and bigmem"` |
| `vllm` | Boots the real engine binary from `.vllm`. | `pytest -m vllm` |

Combine them like any other marker expression:

```bash
pytest -m "gpu and not network"    # GPU tests that use no downloads
pytest -m ""                       # absolutely everything
```

Markers declare *intent*; a second mechanism (`pytest_runtest_setup` in
`tests/conftest.py`) checks the *environment*. A `gpu` test explicitly selected
on a CPU-only box **skips with a reason** rather than erroring, and the same
goes for `vllm` with no engine binary on disk and `network` when
`HF_HUB_OFFLINE=1`.

Every tier, `vllm` included, runs from the **main** venv. The pipeline never
imports `vllm`: `llm_server_context` launches `<vllm_venv_path>/bin/vllm` as a
subprocess, so the tier needs that binary to exist, not the package to be
importable. `.vllm` holds the engine but not the project's own dependencies, so
running the suite from inside it cannot even import `tests/conftest.py`. The
venv is located from `VLLM_VENV_PATH`, then `vllm_venv_path` in `globals.toml`,
then `.vllm`.

## Layout

```
tests/
  conftest.py          tier gating + shared fixtures (tiny models, fake OpenAI server)
  data/                committed fixture files (prompt sets, TOML configs)
  unit/                default tier - logic, config validation, numerical correctness
  integration/         network tier - real Hub datasets and checkpoints
  smoke/               gpu / vllm tier - preflight before a production run
```

Directories are for humans; the markers are what the runner selects on.

## The default tier is genuinely offline

The model-dependent tests do not download anything: `tests/conftest.py` builds
randomly-initialised checkpoints in-process (a two-layer Llama, a two-layer
BERT classifier, and a word-level tokenizer). That is enough to run the real
the real LLM scoring pass, the real EditLens inference loop, and the real `transformers`
code paths — `tests/unit/test_llm_scoring.py` checks every reduction against a
naive per-position reference implementation.

`batch_generate` is likewise tested against a real OpenAI-compatible server
(`fake_openai_server`) bound to localhost, and the engine launch/health-check
loop against a stub `vllm` binary, so the actual client, subprocess, and
polling code runs without a GPU.

## Preflight: catching OOMs before the job, not during it

`tests/smoke/test_gpu_preflight.py` loads the checkpoints named in `config/` and
pushes a deliberately worst-case batch through them, then asserts the measured
peak fits inside a fraction of the card:

```bash
pytest -m "gpu and slow" tests/smoke/test_gpu_preflight.py -s
# [preflight] llm_stats (worst-case batch): peak 21.34 GiB / 24.00 GiB on cuda:0 (89%)
```

`-s` shows the headroom line even when the test passes. Tune the allowance with
`FASTDETECTOR_TEST_VRAM_BUDGET` (default `0.9`).

**What a green preflight does and does not prove.** The synthetic batches are
saturated: every text is padded to the configured sequence cap and there are as
many of them as the batch size allows. That is the shape that OOMs, because
`SentenceTransformer.encode` sorts by length and pads each batch to its longest
member, so the longest rows in a shard get batched together.

It is still synthetic. Passing means *this config fits when saturated*, not
*this shard fits* — token counts, and so activation sizes, depend on the actual
text. To check the data rather than the configuration, point the preflight at a
real shard:

```bash
FASTDETECTOR_TEST_PREFLIGHT_DATASET=G-reen/cc-2021-stat \
  pytest -m "gpu and slow" tests/smoke/test_gpu_preflight.py -s
```

That pulls the longest `FASTDETECTOR_TEST_PREFLIGHT_ROWS` rows (default 8) of
shard `FASTDETECTOR_TEST_PREFLIGHT_SHARD` (default 0) and runs those. It skips
when the variable is unset, so it costs nothing by default.

`tests/smoke/test_pipeline_smoke.py` is the equivalent for generation: it boots
the real engine from `config/gen/train/shard_0.toml` and runs documents through
`run_pipeline` end to end.

## Environment variables

| Variable | Default | Used for |
|---|---|---|
| `FASTDETECTOR_TEST_MODEL` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | checkpoint for the `network` scorer tests |
| `FASTDETECTOR_TEST_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | sentence-embedding tests |
| `FASTDETECTOR_TEST_TOKEN_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | token-embedding tests |
| `FASTDETECTOR_TEST_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | cross-encoder tests |
| `FASTDETECTOR_TEST_HF_DATASET` | *(unset - tests skip)* | Hub dataset read tests |
| `FASTDETECTOR_TEST_HF_WRITE_DATASET` | *(unset - tests skip)* | Hub shard round-trip; **writes real data** |
| `FASTDETECTOR_TEST_EDITLENS` | *(unset - test skips)* | download the real ~1.4GB EditLens checkpoint |
| `FASTDETECTOR_TEST_GEN_CONFIG` | `config/gen/train/shard_0.toml` | which gen config the pipeline smoke test boots |
| `FASTDETECTOR_TEST_FILTER_ENGINE` | *(unset - test skips)* | also boot the filter model |
| `FASTDETECTOR_TEST_VRAM_BUDGET` | `0.9` | allowed fraction of VRAM in the preflight |
| `FASTDETECTOR_TEST_PREFLIGHT_DATASET` | *(unset - test skips)* | real dataset whose longest rows the preflight samples |
| `FASTDETECTOR_TEST_PREFLIGHT_SHARD` | `0` | which shard/batch-id to sample |
| `FASTDETECTOR_TEST_PREFLIGHT_ROWS` | `8` | how many of the longest rows to use |
| `FASTDETECTOR_TEST_CUDA_DEVICE` | `cuda:0` | device the GPU tier reports against |
| `FASTDETECTOR_TEST_OFFLINE` | *(unset)* | set to `1` to skip every `network` test |
| `VLLM_VENV_PATH` | `vllm_venv_path` in `globals.toml`, else `.vllm` | engine venv the `vllm` tier looks for a binary in |

Point the model variables at the production checkpoints to smoke those instead
of the tiny defaults:

```bash
FASTDETECTOR_TEST_MODEL=unsloth/Llama-3.2-3B-Instruct pytest -m network
```

## What the config tests cover

`tests/unit/test_repo_configs.py` runs against the **real** files in `config/`
and `prompts/`, not fixtures. It catches, in about a second, things that would
otherwise surface hours into a run: a sampling parameter the configured engine
silently ignores, a filter operator that would quietly empty the dataset, a
missing prompt file, a gap in the shard numbering, a `max_input_len` larger than
the context window, or stat configs that disagree about which columns to score.

## CI

`.github/workflows/tests.yml` runs the default tier on every push and pull
request, with `HF_HUB_OFFLINE=1` set so an accidentally-downloading "unit" test
fails loudly.

The GPU tiers run on the cluster instead — see `slurm/tests/run_tests.sbatch`,
which runs the `slow`, `network`, and `gpu` tiers in sequence and is the
recommended thing to submit before kicking off a long production job.

## Adding tests

- No marker means it must be fast, offline, and CPU-only. If it is not, mark it.
- Prefer the offline model fixtures (`tiny_lm`, `tiny_tokenizer`,
  `tiny_sequence_classifier`) over downloading; they exercise the same code.
- `--strict-markers` is on: a typo'd marker is a collection error, not a test
  that silently never runs.
