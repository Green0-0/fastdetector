FastDetector works through five general pipeline stages:

0. **Writing prompts:** sample prompts are provided in sample_prompts/, but you are free to write your own. Once you have a list of prompts you should build (or have an agent build) them into a dataset, reference scripts/prompts and src/fastdetector/prompting. Finished example datasets are placed inside prompts/.
1. **Filtering data:** we want to construct a reasonably clean, boilerplate-free human written dataset. This can be done by passing the text through an LLM.
2. **Generating data:** now that we have our human data, we need an AI example to go with each. In other words:

    **[vLLM](https://github.com/vllm-project/vllm) go!**
    * [Aphrodite](https://github.com/dphnAI/sonar) is also supported if special samplers like XTC are desired.
3. **Calculating stats:** we want to know a lot of things about our AI data. How different is it from our human data? What score do AI text detectors give it?
4. **Analysis / Readme building:** with our data calculated, we can automatically generate a comprehensive report and push it to huggingface.

---

These four stages (ignoring prompts) correspond to the following python scripts (in scripts) and configs (in config):

**#1:** filter.py and filter.toml

**#2:** gen.py and gen/shard_N.toml

**#3:** stats/distance_stats.py and distance_stats.toml, stats/editlens_stats.py and editlens_stats.toml, stats/llm_stats.py and llm_stats.toml

**#4:** analysis.py and analysis.toml

There is also a globals.toml which specifies a naming convention for  huggingface datasets; **you should modify it with your username; you also need the source dataset under your account**.

The configuration parameters are reasonably straightforward, consult your favorite agent or raise an issue for help if absolutely necessary.

