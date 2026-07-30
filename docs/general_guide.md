FastDetector works through six general pipeline stages:

0. **Writing prompts:** sample prompts are provided in sample_prompts/, but you are free to write your own. Once you have a list of prompts you should build (or have an agent build) them into a dataset, reference scripts/prompts and src/fastdetector/prompting. Finished example datasets are placed inside prompts/.
1. **Sharding data:** split your raw dataset into the shards the rest of the pipeline processes. This is also where you choose how much of the corpus to use.
2. **Filtering data:** we want to construct a reasonably clean, boilerplate-free human written dataset. This can be done by passing the text through an LLM.
3. **Generating data:** now that we have our human data, we need an AI example to go with each. In other words:

    **[vLLM](https://github.com/vllm-project/vllm) go!**
    * [Aphrodite](https://github.com/dphnAI/sonar) is also supported if special samplers like XTC are desired.
4. **Calculating stats:** we want to know a lot of things about our AI data. How different is it from our human data? What score do AI text detectors give it?
5. **Analysis / Readme building:** with our data calculated, we can automatically generate a comprehensive report and push it to huggingface.

---

These five stages (ignoring prompts) correspond to the following python scripts (in scripts) and configs (in config):

**#1:** shard_dataset.py, configured by its arguments rather than a TOML: `uv run scripts/shard_dataset.py --num-shards 8 --num-samples 5000000`

**#2:** filter.py and filter.toml

**#3:** gen.py and gen/shard_N.toml

**#4:** stats/distance_stats.py and distance_stats.toml, stats/editlens_stats.py and editlens_stats.toml, stats/llm_stats.py and llm_stats.toml

**#5:** analysis.py and analysis.toml

There is also a globals.toml which specifies traditional dataset paths and an optional username prefix for huggingface datasets; **you should modify it with your dataset paths (and optional prefix); you also need the source dataset under your account**.

Stages #2 to #4 take a `--batch-id`: each one processes the shard with that index and writes its results back under the same shard name, so scaling out is one batch-id per machine (and stage #5 reads every shard back). Nothing else decides how much data a run covers — no config file carries a sample count, and each stage processes every row of the shard it is handed, so `--num-samples` at stage #1 is the single place that is set.

`slurm/` holds a job script per stage, with the sharded ones submitted as array jobs (their `--array` range and your `--num-shards` have to agree). If you need to start over, `scripts/delete_datasets.py` deletes the datasets a set of stages wrote (it only lists them unless you pass `--yes`).

Every GPU stage points its compile caches at node-local disk (`/tmp/fd_$SLURM_JOB_ID`) and deletes them on exit. This is not optional on a cluster whose `$HOME` is NFS: Triton otherwise writes to `$HOME/.triton`, and several array tasks compiling the same kernels into that one shared directory race each other until NFS returns `ESTALE` — which shows up as `Triton compilation failed` / `OSError: [Errno 116] Stale file handle` and kills the worker, looking for all the world like a CUDA problem. It also keeps compile artifacts out of a quota-limited home directory. If your `/tmp` is small or not node-local, override `FD_CACHE_ROOT` in the job scripts.

The configuration parameters are reasonably straightforward, consult your favorite agent or raise an issue for help if absolutely necessary.

One worth knowing about is `analysis.toml`, because it decides what the report contains: a statistic your stats stage computed but that is not listed there as a `distance_metrics` entry or a `[[classifiers]]` block simply never appears in the readme. The committed file lists everything the pipeline can compute — every distance metric, the EditLens score/bucket, and the LLM detectors (perplexity, entropy, top-p/top-k outliers, FastDetectGPT and Binoculars, per checkpoint) — and naming one you have not computed is safe: analysis.py evaluates what the dataset actually has and reports the rest as skipped, so a stage you have not run is visible in the readme rather than silently missing. Each classifier's `direction` says which side of the threshold is AI (`lower_is_ai` for perplexity, entropy, outlier rates and Binoculars; `higher_is_ai` for the rest); setting it wrongly inverts that classifier's AUROC. The classifier suffixes have to track the stats configs — `col_suffixes` in `llm_stats.toml` and `suffix` in `editlens_stats.toml` — and `uv run pytest tests/unit/test_repo_configs.py` checks that they still line up.

One that is worth knowing about explicitly: `distance_stats.toml` sets `embedding_max_seq_length` and `reranker_max_length` (both `8192` by default). Left unset, the Qwen3 embedding and reranker passes inherit the checkpoints' own 40960-token limit, and because `SentenceTransformer.encode` sorts inputs by length and pads each batch to its longest member, a single runaway generation drags its whole batch's memory up with it — which is how this stage OOMs on a 24 GiB card. Raise the caps if you have the VRAM, but note this is a semantic choice as well as a memory one: text beyond the cap is truncated and so does not contribute to the metric.

