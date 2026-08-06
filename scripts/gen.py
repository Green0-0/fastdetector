import argparse
import re

from fastdetector.frontend.toml_config import GenConfig
from fastdetector.frontend.toml_loader import load_config_pair
from fastdetector.frontend.pipe import run_pipeline
from fastdetector.statistics.filters import (
    fix_encoding,
    has_filler_output,
    has_meta_commentary,
    has_placeholder,
    has_prompt_echo,
    has_refusal,
    is_empty,
    normalize_whitespace,
    strip_added_title,
    strip_emoji,
    strip_markdown,
    strip_wrapper_boilerplate,
)
from fastdetector.utils import push_shard, shard_config_name, upload_readme

#: Template placeholders the prompt column keeps unsubstituted.
PLACEHOLDER_TOKENS = re.compile(r"\{\{(?:DOC|TEXT|RESP_\d+)\}\}")


def prompt_instructions(prompts: list[dict]) -> list[str]:
    """Recover the instruction text behind each row, without the document.

    The prompt column stores the template rather than the substituted prompt, so
    dropping the placeholders leaves the instruction on its own.

    Args:
        prompts: List of prompt metadata dicts carrying ``chat_turns``.

    Returns:
        List of instruction texts, one per row.
    """
    return [PLACEHOLDER_TOKENS.sub("", "\n".join((prompt or {}).get("chat_turns") or [])) for prompt in prompts]


def clean_columns(originals: list[str], responses: list[str]) -> tuple[list[str], list[str]]:
    """Repair and normalize the human and AI text columns.

    Wrapper and title stripping run first, against the untouched source text,
    because both decide what to remove by comparing the two columns. Markdown,
    emoji and whitespace are then normalized on *both* columns: each is a
    generator fingerprint rather than a property of AI text, and the source
    column arrives pre-normalized by the extractor, so cleaning only one side
    would let formatting alone identify the AI side.

    Args:
        originals: List of source texts.
        responses: List of model responses aligned with ``originals``.

    Returns:
        Tuple of (cleaned originals, cleaned responses).
    """
    originals = fix_encoding(originals)
    responses = strip_wrapper_boilerplate(fix_encoding(responses), originals)
    responses = strip_added_title(responses, originals)
    return (normalize_whitespace(strip_emoji(strip_markdown(originals))),
            normalize_whitespace(strip_emoji(strip_markdown(responses))))


def rejected_rows(originals: list[str], responses: list[str],
                  instructions: list[str]) -> tuple[list[bool], dict[str, int]]:
    """Flag rows whose generation failed outright.

    Similarity between the two columns is deliberately not considered here; the
    analysis stage drops over-similar pairs via its own filter conditions.

    Args:
        originals: List of source texts.
        responses: List of model responses aligned with ``originals``.
        instructions: List of instruction texts aligned with ``originals``.

    Returns:
        Tuple of (per-row rejection flags, per-reason counts).
    """
    reasons = {
        "empty or too short": is_empty(responses),
        "refusal": has_refusal(responses),
        "filler output": has_filler_output(responses, originals),
        "unfilled placeholder": has_placeholder(responses, originals),
        "task meta-commentary": has_meta_commentary(responses, originals),
        "echoed instruction": has_prompt_echo(responses, instructions),
    }
    return [any(flags) for flags in zip(*reasons.values())], {k: sum(v) for k, v in reasons.items()}


def main() -> None:
    """Execute the text generation pipeline from command line configuration.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Run the LLM Generation pipeline.")
    parser.add_argument("--globals-config", type=str, default="config/globals.toml", help="Path to globals.toml")
    parser.add_argument("--gen-config", type=str, required=True, help="Path to gen TOML (e.g. config/gen/shard_0.toml)")
    parser.add_argument("--batch-id", type=int, default=0, help="Batch ID to automatically pick a subset of the dataset.")

    args = parser.parse_args()

    globals_config, task_config = load_config_pair(
        args.globals_config, args.gen_config, GenConfig
    )

    source_dataset = globals_config.resolve_dataset(globals_config.post_filter_dataset)
    target_dataset = globals_config.resolve_dataset(globals_config.gen_dataset)
    stat_dataset = globals_config.resolve_dataset(globals_config.stat_dataset)

    print("Running generation pipeline...")
    print(f"Source Dataset: {source_dataset}")
    print(f"Target Dataset: {target_dataset}")
    print(f"Engine: {task_config.pipeline.engine}")

    result_ds, readme_content = run_pipeline(
        globals_config=globals_config,
        pipe_config=task_config.pipeline,
        prompt_file=task_config.prompt_file,
        source_column=task_config.source_column,
        source_dataset_name=source_dataset,
        batch_id=args.batch_id
    )

    print("Running post-processing...")
    originals, responses = clean_columns(result_ds["original"], result_ds["final_response"])
    rejected, counts = rejected_rows(originals, responses, prompt_instructions(result_ds["prompt"]))

    total = len(result_ds)
    result_ds = result_ds.remove_columns(["original", "final_response"])
    result_ds = result_ds.add_column("original", originals).add_column("final_response", responses)
    result_ds = result_ds.select([i for i, drop in enumerate(rejected) if not drop])

    dropped = "\n".join(f"- Dropped ({reason}): {count}" for reason, count in counts.items())
    readme_content += f"""

## Post Processing Stats
- Rows in: {total}
- Rows kept: {len(result_ds)}
{dropped}

Reasons overlap, so they sum to more than the number of rows dropped. Pairs whose
two sides are too similar are filtered downstream by the analysis stage, not here.
"""

    config_name = shard_config_name(args.batch_id)

    print(f"Pushing dataset to '{target_dataset}' (config '{config_name}')...")
    push_shard(result_ds, target_dataset, config_name=config_name)
    upload_readme(dataset_name=target_dataset, readme_content=readme_content)

    print(f"Pushing cloned dataset to '{stat_dataset}' (config '{config_name}') with a stub readme...")
    push_shard(result_ds, stat_dataset, config_name=config_name)
    stub_readme = "# WIP Fastdetector dataset\nWaiting for statistics to finish generating...\n"
    upload_readme(dataset_name=stat_dataset, readme_content=stub_readme)


if __name__ == "__main__":
    main()
