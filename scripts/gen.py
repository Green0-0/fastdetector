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
    strip_wrapper_boilerplate,
)
from fastdetector.utils import push_shard, shard_config_name, upload_readme


def clean_responses(responses: list[str], originals: list[str]) -> list[str]:
    """Clean the AI column. The human column is read but never modified.

    Args:
        responses: List of model responses to clean.
        originals: List of source texts aligned with ``responses``, used only to
            decide what counts as boilerplate.

    Returns:
        List of cleaned responses.
    """
    responses = strip_wrapper_boilerplate(fix_encoding(responses), originals)
    return normalize_whitespace(strip_added_title(responses, originals))


def rejected_rows(originals: list[str], responses: list[str],
                  instructions: list[str]) -> tuple[list[bool], dict[str, int]]:
    """Flag rows whose generation failed outright.

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
    instructions = [re.sub(r"\{\{(?:DOC|TEXT|RESP_\d+)\}\}", "", "\n".join(prompt["chat_turns"]))
                    for prompt in result_ds["prompt"]]
    originals = result_ds["original"]
    responses = clean_responses(result_ds["final_response"], originals)
    rejected, counts = rejected_rows(originals, responses, instructions)

    total = len(result_ds)
    result_ds = result_ds.remove_columns("final_response").add_column("final_response", responses)
    result_ds = result_ds.select([i for i, drop in enumerate(rejected) if not drop])

    dropped = "\n".join(f"- Dropped ({reason}): {count}" for reason, count in counts.items())
    readme_content += f"""

## Post Processing Stats
- Rows in: {total}
- Rows kept: {len(result_ds)}
{dropped}

Note: Reasons overlap, so they sum to more than the number of rows dropped.
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
