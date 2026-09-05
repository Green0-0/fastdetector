import os

import pytest

from fastdetector.frontend.engine_config import EngineConfig
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.frontend.toml_config import FilterConfig, GenConfig, GlobalsConfig
from fastdetector.frontend.toml_loader import load_toml
from fastdetector.generator import batch_generate
from fastdetector.llm_utils import llm_server_context
from fastdetector.prompting.prompts import PromptSet, load_prompts

pytestmark = [pytest.mark.vllm, pytest.mark.gpu, pytest.mark.slow, pytest.mark.network]

SAMPLE_DOCUMENTS = [
    "The committee approved the proposal after a long and careful debate. "
    "The minutes were circulated the following morning, and no objections "
    "were raised by any of the attending members.",
    "Volcanic activity on the island resumed early yesterday morning, "
    "prompting the local authority to close two coastal roads while the "
    "situation was assessed.",
]


@pytest.fixture(scope="module")
def globals_config(repo_root) -> GlobalsConfig:
    """The repository's globals.toml."""
    return GlobalsConfig(**load_toml(str(repo_root / "config" / "globals.toml")))


@pytest.fixture(scope="module")
def gen_config(repo_root) -> GenConfig:
    """The generation config under test (default: shard 0)."""
    name = os.environ.get("FASTDETECTOR_TEST_GEN_CONFIG", "config/gen/train/shard_0.toml")
    path = repo_root / name
    if not path.is_file():
        pytest.skip(f"{name} does not exist")
    return GenConfig(**load_toml(str(path)))


def test_the_configured_engine_serves_a_completion(repo_root, globals_config, gen_config):
    """Launch the real engine and get one completion out of it."""
    pipeline = gen_config.pipeline
    if not pipeline.engine.is_local_server:
        pytest.skip(f"{pipeline.engine.value} is not a locally launched engine")

    venv_path = (
        globals_config.vllm_venv_path
        if pipeline.engine == EngineConfig.VLLM
        else globals_config.aphrodite_venv_path
    )
    server_kwargs = {}
    if pipeline.max_model_len is not None:
        server_kwargs["max_model_len"] = pipeline.max_model_len

    with llm_server_context(
        engine=pipeline.engine,
        model_name=pipeline.model_name,
        venv_path=venv_path,
        parallelization_type=pipeline.parallelization_type,
        max_num_seqs=pipeline.max_num_seqs,
        max_num_batched_tokens=pipeline.max_num_batched_tokens,
        **server_kwargs,
    ) as api_url:
        texts, prompt_tokens, completion_tokens, failed = batch_generate(
            api_url,
            [[{"role": "user", "content": "Reply with the single word: ready"}]],
            {"temperature": 0.0},
            model_name=pipeline.model_name,
        )

    assert failed == 0, "the engine accepted the request but returned no completion"
    assert texts[0].strip()
    assert prompt_tokens > 0
    assert completion_tokens > 0


def test_the_prompt_file_renders_against_the_real_model(repo_root, gen_config):
    """Every configured prompt must produce a non-empty first user message."""
    prompts = PromptSet(load_prompts([str(repo_root / gen_config.prompt_file)]))
    mapped, labels = prompts.map(SAMPLE_DOCUMENTS)

    assert len(mapped) == len(SAMPLE_DOCUMENTS)
    for prompt, document in zip(mapped, SAMPLE_DOCUMENTS):
        assert document in prompt.chat_turns[0]
        assert "{{DOC}}" not in prompt.chat_turns[0]
    assert all(label["metadata"] for label in labels)


def test_run_pipeline_produces_a_dataset(monkeypatch, repo_root, globals_config, gen_config):
    """The full generation path: engine launch, prompting, and post-processing."""
    from fastdetector.statistics.filters import strip_wrapper_boilerplate

    if not gen_config.pipeline.engine.is_local_server:
        pytest.skip(f"{gen_config.pipeline.engine.value} is not a local engine")

    # Feed the pipeline a couple of in-memory rows instead of the Hub dataset,
    # so the smoke test does not depend on a previous stage having run.
    source_column = gen_config.source_column
    rows = [{source_column: document} for document in SAMPLE_DOCUMENTS]
    monkeypatch.setattr(
        "fastdetector.frontend.pipe.load_dataset_auto_shard",
        lambda *args, **kwargs: rows,
    )

    dataset, readme = run_pipeline(
        globals_config=globals_config,
        pipe_config=gen_config.pipeline,
        prompt_file=str(repo_root / gen_config.prompt_file),
        num_samples=len(rows),
        source_column=source_column,
        source_dataset_name="in-memory-smoke-test",
        batch_id=0,
    )

    assert len(dataset) == len(rows)
    assert "final_response" in dataset.column_names
    assert "generator_model" in dataset.column_names
    assert all(text.strip() for text in dataset["final_response"])
    assert "Failed API Requests: 0" in readme

    processed = strip_wrapper_boilerplate(dataset["final_response"], dataset["original"])
    assert all(text.strip() for text in processed)


def test_the_filter_config_engine_also_serves(repo_root, globals_config):
    """The filtering stage uses its own model; check that one boots too."""
    if os.environ.get("FASTDETECTOR_TEST_FILTER_ENGINE") != "1":
        pytest.skip(
            "set FASTDETECTOR_TEST_FILTER_ENGINE=1 to also boot the filter model"
        )

    config = FilterConfig(**load_toml(str(repo_root / "config" / "filter.toml")))
    pipeline = config.pipeline
    if not pipeline.engine.is_local_server:
        pytest.skip(f"{pipeline.engine.value} is not a locally launched engine")

    venv_path = (
        globals_config.vllm_venv_path
        if pipeline.engine == EngineConfig.VLLM
        else globals_config.aphrodite_venv_path
    )
    server_kwargs = {}
    if pipeline.max_model_len is not None:
        server_kwargs["max_model_len"] = pipeline.max_model_len

    with llm_server_context(
        engine=pipeline.engine,
        model_name=pipeline.model_name,
        venv_path=venv_path,
        parallelization_type=pipeline.parallelization_type,
        max_num_seqs=pipeline.max_num_seqs,
        max_num_batched_tokens=pipeline.max_num_batched_tokens,
        **server_kwargs,
    ) as api_url:
        texts, _, _, failed = batch_generate(
            api_url,
            [[{"role": "user", "content": "Reply with the single word: ready"}]],
            {"temperature": 0.0},
            model_name=pipeline.model_name,
        )
    assert failed == 0
    assert texts[0].strip()
