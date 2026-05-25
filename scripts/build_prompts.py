import random

from fastdetector.prompt_builder import (
    shuffle, resize, partial_stack, force_reformat,
    apply_recursive_format, load_raw_samples, generate_dataset, save_dataset,
)

def build_prompts_generic(paths, dataset_name, target_size, max_stack):
    """Load, resize, stack, format with recursive headers, and save."""
    samples = load_raw_samples(paths)
    print(f"Building {dataset_name} using {len(samples)} sample prompts from {len(paths)} files...")
    samples = resize(samples, target_size)
    copies = [shuffle(samples, seed=i) for i in range(max_stack)]
    samples = partial_stack(copies, 1, max_stack)
    samples = force_reformat(samples, only_first_message=True, modified_format="{{DOC}}\n{{TEXT}}")
    samples = force_reformat(samples, only_first_message=False, modified_format="{{TEXT}}\nOutput the full new text with no extra statements or commentations.")
    samples = apply_recursive_format(samples)
    prompts = generate_dataset(samples, use_multiturn=False)
    save_dataset(prompts, dataset_name)
    print(f"  Saved {len(prompts)} prompts to {dataset_name}")
    return prompts

def build_indirect_reference(subcategories, dataset_name, target_size):
    print(f"Building {dataset_name} using {len(subcategories)} files...")
    per_file = target_size // len(subcategories)
    all_prompts = []
    for path, followup_text in subcategories.items():
        samples = load_raw_samples([path])
        print(f"  Loaded {len(samples)} samples from {path}")
        samples = resize(samples, per_file)
        samples = force_reformat(samples, only_first_message=True, modified_format="{{DOC}}\n{{TEXT}}\nDo not output anything besides what you were requested to write, and do not output any extra commentary.")
        samples = [chat + [followup_text] for chat in samples]
        samples = apply_recursive_format(samples)
        prompts = generate_dataset(samples, use_multiturn=False)
        all_prompts.extend(prompts)
    save_dataset(all_prompts, dataset_name)
    print(f"  Saved {len(all_prompts)} prompts to {dataset_name}")
    return all_prompts

def main():
    all_prompts = []

    all_prompts += build_prompts_generic([
        "sample_prompts/direct_reference/adversarial.json",
        "sample_prompts/direct_reference/situation.json",
        "sample_prompts/direct_reference/style.json",
    ], "direct_reference_dataset", target_size=500, max_stack=1)

    all_prompts += build_prompts_generic([
        "sample_prompts/revise/audience.json",
        "sample_prompts/revise/clarify.json",
        "sample_prompts/revise/edit.json",
        "sample_prompts/revise/elaboration.json",
        "sample_prompts/revise/restructure.json",
        "sample_prompts/revise/tone.json",
    ], "revise_dataset", target_size=1500, max_stack=2)

    all_prompts += build_prompts_generic([
        "sample_prompts/rewrite/miscellaneous.json",
        "sample_prompts/rewrite/section.json",
        "sample_prompts/rewrite/sentence.json",
        "sample_prompts/rewrite/word.json",
    ], "rewrite_dataset", target_size=2500, max_stack=2)

    all_prompts += build_indirect_reference({
        "sample_prompts/indirect_reference/descriptive_encode.json":
            "Above is an AI generated descriptor/trace of some human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/partial_encode.json":
            "Above is a partial trace of a human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/prompt_encode.json":
            "Output only the generated text with no extra commentary.",
        "sample_prompts/indirect_reference/translation_roundtrip.json":
            "Translate this text to English. Output only the English translation with no extra commentary.",
    }, "indirect_reference_dataset", target_size=500)

    random.seed(42)
    random.shuffle(all_prompts)
    save_dataset(all_prompts, "combined_dataset")
    print(f"\nSaved {len(all_prompts)} combined prompts to combined_dataset")
    print("All datasets built successfully.")


if __name__ == "__main__":
    main()
