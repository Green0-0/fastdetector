import random

from fastdetector.prompting.prompt_builder import (
    add_final_instruction_variants,
    add_metadata,
    apply_recursive_format,
    force_reformat,
    generate_dataset,
    load_raw_samples,
    partial_stack,
    resize,
    shuffle,
    save_dataset,
)
from fastdetector.prompting.prompts import Prompt


TOTAL_PROMPTS = 150_000
METADATA_INSTRUCTIONS = [
    None,
    "The final text must have the topic <<topic>>.",
    "The final text must have the format <<format>>.",
    "The final text must have the format <<format>> and topic <<topic>>.",
]


def build_prompts_generic(
    paths: list[str],
    dataset_name: str,
    prompt_type: str,
    target_size: int,
    max_stack: int,
) -> list[Prompt]:
    """Load, resize, stack, and format one generic prompt family.

    Args:
        paths: List of raw sample JSON file paths.
        dataset_name: Name prefix for saved dataset files.
        prompt_type: PROMPT_TYPE metadata string.
        target_size: Number of prompts to generate.
        max_stack: Maximum number of samples to stack per prompt.

    Returns:
        Generated prompts.
    """
    samples = load_raw_samples(paths)
    print(
        f"Building {dataset_name} using {len(samples)} sample prompts "
        f"from {len(paths)} files..."
    )
    samples = resize(samples, target_size)
    copies = [shuffle(samples, seed=i) for i in range(max_stack)]
    samples = partial_stack(copies, 1, max_stack)
    samples = force_reformat(
        samples,
        only_first_message=True,
        modified_format="<document>\n{{DOC}}\n</document>\n\n{{TEXT}}",
    )
    samples = force_reformat(
        samples,
        only_first_message=False,
        modified_format=(
            "{{TEXT}}\nOutput the full new text with no extra statements or "
            "commentations.\nBegin directly with the text itself. Do not add a "
            "title, a heading, or a label naming what you have written."
        ),
    )
    samples = apply_recursive_format(samples)
    prompts = generate_dataset(samples, use_multiturn=False)
    add_metadata(prompts, "PROMPT_TYPE", prompt_type)
    return prompts


def build_indirect_reference(
    subcategories: dict[str, str],
    dataset_name: str,
    prompt_type: str,
    target_size: int,
) -> list[Prompt]:
    """Build one indirect-reference prompt family from all source variants.

    Args:
        subcategories: Dictionary mapping file path to follow-up instruction text.
        dataset_name: Name prefix for saved dataset files.
        prompt_type: PROMPT_TYPE metadata string.
        target_size: Total target dataset size.

    Returns:
        Generated prompts.
    """
    print(f"Building {dataset_name} using {len(subcategories)} files...")
    per_file, remainder = divmod(target_size, len(subcategories))
    all_prompts: list[Prompt] = []

    for index, (path, followup_text) in enumerate(subcategories.items()):
        samples = load_raw_samples([path])
        size = per_file + (1 if index < remainder else 0)
        print(f"  Loaded {len(samples)} samples from {path}; generating {size}")
        samples = resize(samples, size)
        samples = force_reformat(
            samples,
            only_first_message=True,
            modified_format=(
                "<document>\n{{DOC}}\n</document>\n\n{{TEXT}}\nDo not output "
                "anything besides what you were requested to write, and do not "
                "output any extra commentary."
            ),
        )
        samples = [
            chat + [
                f"{followup_text}\nBegin directly with the text itself. Do not "
                "add a title, a heading, or a label naming what you have written."
            ]
            for chat in samples
        ]
        samples = apply_recursive_format(samples)
        prompts = generate_dataset(samples, use_multiturn=False)
        add_metadata(prompts, "PROMPT_TYPE", prompt_type)
        all_prompts.extend(prompts)

    return all_prompts


def main() -> None:
    """Build all prompt datasets (direct, revise, rewrite, indirect) and write outputs to disk.

    Returns:
        None.
    """
    all_prompts: list[Prompt] = []
    family_size = TOTAL_PROMPTS // 4

    all_prompts.extend(build_prompts_generic([
        "sample_prompts/direct_reference/adversarial.json",
        "sample_prompts/direct_reference/situation.json",
        "sample_prompts/direct_reference/style.json",
    ], "direct_reference", "direct_reference", target_size=family_size, max_stack=1))

    all_prompts.extend(build_prompts_generic([
        "sample_prompts/revise/audience.json",
        "sample_prompts/revise/clarify.json",
        "sample_prompts/revise/edit.json",
        "sample_prompts/revise/elaboration.json",
        "sample_prompts/revise/restructure.json",
        "sample_prompts/revise/tone.json",
    ], "revise", "revise", target_size=family_size, max_stack=2))

    all_prompts.extend(build_prompts_generic([
        "sample_prompts/rewrite/miscellaneous.json",
        "sample_prompts/rewrite/section.json",
        "sample_prompts/rewrite/sentence.json",
        "sample_prompts/rewrite/word.json",
    ], "rewrite", "rewrite", target_size=family_size, max_stack=2))

    all_prompts.extend(build_indirect_reference({
        "sample_prompts/indirect_reference/descriptive_encode.json":
            "Above is an AI generated descriptor/trace of some human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible, noting that many details have been lost/excluded in the descriptor, and you must expand upon it to recover the original text. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/partial_encode.json":
            "Above is a partial trace of a human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible, noting that many details have been lost/excluded in the descriptor, and you must expand upon it to recover the original text. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/prompt_encode.json":
            "Output only the generated text with no extra commentary.",
        "sample_prompts/indirect_reference/translation_roundtrip.json":
            "Translate this text to English. Output only the English translation with no extra commentary.",
    }, "indirect_reference", "indirect_reference", target_size=family_size))

    assert len(all_prompts) == TOTAL_PROMPTS
    add_final_instruction_variants(all_prompts, METADATA_INSTRUCTIONS, seed=42)
    random.Random(42).shuffle(all_prompts)
    save_dataset(all_prompts, "combined_dataset")

    print(f"\nSaved {len(all_prompts)} prompts to combined_dataset")
    print("All datasets built successfully.")


if __name__ == "__main__":
    main()
