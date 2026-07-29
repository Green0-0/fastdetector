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

There is also a globals.toml which specifies a naming convention for  huggingface datasets; **you should modify it with your username; you also need the source dataset under your account**.

Stages #2 to #4 take a `--batch-id`: each one processes the shard with that index and writes its results back under the same shard name, so scaling out is one batch-id per machine (and stage #5 reads every shard back). Nothing else decides how much data a run covers — no config file carries a sample count, and each stage processes every row of the shard it is handed, so `--num-samples` at stage #1 is the single place that is set.

`slurm/` holds a job script per stage, with the sharded ones submitted as array jobs (their `--array` range and your `--num-shards` have to agree). If you need to start over, `scripts/delete_datasets.py` deletes the datasets a set of stages wrote (it only lists them unless you pass `--yes`).

Every GPU stage points its compile caches at node-local disk (`/tmp/fd_$SLURM_JOB_ID`) and deletes them on exit. This is not optional on a cluster whose `$HOME` is NFS: Triton otherwise writes to `$HOME/.triton`, and several array tasks compiling the same kernels into that one shared directory race each other until NFS returns `ESTALE` — which shows up as `Triton compilation failed` / `OSError: [Errno 116] Stale file handle` and kills the worker, looking for all the world like a CUDA problem. It also keeps compile artifacts out of a quota-limited home directory. If your `/tmp` is small or not node-local, override `FD_CACHE_ROOT` in the job scripts.

The configuration parameters are reasonably straightforward, consult your favorite agent or raise an issue for help if absolutely necessary.

