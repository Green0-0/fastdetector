import random

from fastdetector.prompting.prompt_builder import (
    add_metadata,
    apply_recursive_format,
    force_reformat,
    generate_dataset,
    load_raw_samples_balanced_autosplit,
    partial_stack,
    resize,
    save_dataset,
    shuffle,
)

def build_prompts_generic(paths: list[str], dataset_name: str, prompt_type: str, target_size: int, max_stack: int) -> tuple[list, list]:
    """Load, split, resize, stack, format with recursive headers, and save generic prompt sets.

    Args:
        paths: List of raw sample JSON file paths.
        dataset_name: Name prefix for saved dataset files.
        prompt_type: PROMPT_TYPE metadata string.
        target_size: Total number of prompts to generate across train and test splits.
        max_stack: Maximum number of samples to stack per prompt.

    Returns:
        Tuple of (train_prompts_list, test_prompts_list).
    """
    train_samples, test_samples = load_raw_samples_balanced_autosplit(paths, split_proportion=0.8, min_size=1, shuffle_before_split=True)
    
    train_size = int(target_size * 0.8)
    test_size = int(target_size * 0.2)
    
    def process(samples: list, name: str, size: int) -> list:
        """Process, format, and save a subset of samples into Prompt dataclasses.

        Args:
            samples: List of raw sample chats.
            name: Dataset filename to save.
            size: Target number of prompts.

        Returns:
            List of generated Prompt objects.
        """
        print(f"Building {name} using {len(samples)} sample prompts from {len(paths)} files...")
        samples = resize(samples, size)
        copies = [shuffle(samples, seed=i) for i in range(max_stack)]
        samples = partial_stack(copies, 1, max_stack)
        samples = force_reformat(samples, only_first_message=True, modified_format="<document>\n{{DOC}}\n</document>\n\n{{TEXT}}")
        samples = force_reformat(samples, only_first_message=False, modified_format="{{TEXT}}\nOutput the full new text with no extra statements or commentations.\nBegin directly with the text itself. Do not add a title, a heading, or a label naming what you have written.")
        samples = apply_recursive_format(samples)
        prompts = generate_dataset(samples, use_multiturn=False)
        add_metadata(prompts, "PROMPT_TYPE", prompt_type)
        save_dataset(prompts, name)
        print(f"  Saved {len(prompts)} prompts to {name}")
        return prompts

    train_prompts = process(train_samples, f"{dataset_name}_train", train_size)
    test_prompts = process(test_samples, f"{dataset_name}_test", test_size)

    return train_prompts, test_prompts


def build_indirect_reference(subcategories: dict[str, str], dataset_name: str, prompt_type: str, target_size: int) -> tuple[list, list]:
    """Build and save indirect reference prompt datasets from subcategory mapping.

    Args:
        subcategories: Dictionary mapping file path to follow-up instruction text.
        dataset_name: Name prefix for saved dataset files.
        prompt_type: PROMPT_TYPE metadata string.
        target_size: Total target dataset size.

    Returns:
        Tuple of (train_prompts_list, test_prompts_list).
    """
    print(f"Building {dataset_name} using {len(subcategories)} files...")
    train_target_size = int(target_size * 0.8)
    test_target_size = int(target_size * 0.2)
    
    per_file_train = train_target_size // len(subcategories)
    per_file_test = test_target_size // len(subcategories)
    
    all_train_prompts = []
    all_test_prompts = []
    
    for path, followup_text in subcategories.items():
        train_samples, test_samples = load_raw_samples_balanced_autosplit([path], split_proportion=0.8, min_size=1, shuffle_before_split=True)
        print(f"  Loaded {len(train_samples)} train and {len(test_samples)} test samples from {path}")
        
        def process(samples: list, size: int) -> list:
            """Process and format indirect reference samples.

            Args:
                samples: List of raw sample chats.
                size: Target number of prompts.

            Returns:
                List of generated Prompt objects.
            """
            samples = resize(samples, size)
            samples = force_reformat(samples, only_first_message=True, modified_format="<document>\n{{DOC}}\n</document>\n\n{{TEXT}}\nDo not output anything besides what you were requested to write, and do not output any extra commentary.")
            samples = [chat + [f"{followup_text}\nBegin directly with the text itself. Do not add a title, a heading, or a label naming what you have written."] for chat in samples]
            samples = apply_recursive_format(samples)
            prompts = generate_dataset(samples, use_multiturn=False)
            add_metadata(prompts, "PROMPT_TYPE", prompt_type)
            return prompts

        all_train_prompts.extend(process(train_samples, per_file_train))
        all_test_prompts.extend(process(test_samples, per_file_test))
        
    save_dataset(all_train_prompts, f"{dataset_name}_train")
    save_dataset(all_test_prompts, f"{dataset_name}_test")
    print(f"  Saved {len(all_train_prompts)} train and {len(all_test_prompts)} test prompts to {dataset_name}")
    return all_train_prompts, all_test_prompts


def main() -> None:
    """Build all prompt datasets (direct, revise, rewrite, indirect) and write outputs to disk.

    Returns:
        None.
    """
    all_train_prompts = []
    all_test_prompts = []

    train_prompts, test_prompts = build_prompts_generic([
        "sample_prompts/direct_reference/adversarial.json",
        "sample_prompts/direct_reference/situation.json",
        "sample_prompts/direct_reference/style.json",
    ], "direct_reference_dataset", "direct_reference", target_size=2500, max_stack=1)
    all_train_prompts.extend(train_prompts)
    all_test_prompts.extend(test_prompts)

    train_prompts, test_prompts = build_prompts_generic([
        "sample_prompts/revise/audience.json",
        "sample_prompts/revise/clarify.json",
        "sample_prompts/revise/edit.json",
        "sample_prompts/revise/elaboration.json",
        "sample_prompts/revise/restructure.json",
        "sample_prompts/revise/tone.json",
    ], "revise_dataset", "revise", target_size=2500, max_stack=2)
    all_train_prompts.extend(train_prompts)
    all_test_prompts.extend(test_prompts)

    train_prompts, test_prompts = build_prompts_generic([
        "sample_prompts/rewrite/miscellaneous.json",
        "sample_prompts/rewrite/section.json",
        "sample_prompts/rewrite/sentence.json",
        "sample_prompts/rewrite/word.json",
    ], "rewrite_dataset", "rewrite", target_size=2500, max_stack=2)
    all_train_prompts.extend(train_prompts)
    all_test_prompts.extend(test_prompts)

    train_prompts, test_prompts = build_indirect_reference({
        "sample_prompts/indirect_reference/descriptive_encode.json":
            "Above is an AI generated descriptor/trace of some human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible, noting that many details have been lost/excluded in the descriptor, and you must expand upon it to recover the original text. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/partial_encode.json":
            "Above is a partial trace of a human written document, in some arbitrary format. Based on that descriptor, recreate the original human written text it describes as accurately as possible, noting that many details have been lost/excluded in the descriptor, and you must expand upon it to recover the original text. Output only the recreated text with no extra commentary.",
        "sample_prompts/indirect_reference/prompt_encode.json":
            "Output only the generated text with no extra commentary.",
        "sample_prompts/indirect_reference/translation_roundtrip.json":
            "Translate this text to English. Output only the English translation with no extra commentary.",
    }, "indirect_reference_dataset", "indirect_reference", target_size=2500)
    all_train_prompts.extend(train_prompts)
    all_test_prompts.extend(test_prompts)

    random.seed(42)
    random.shuffle(all_train_prompts)
    save_dataset(all_train_prompts, "combined_dataset_train")
    
    random.seed(42)
    random.shuffle(all_test_prompts)
    save_dataset(all_test_prompts, "combined_dataset_test")
    
    print(f"\nSaved {len(all_train_prompts)} combined prompts to combined_dataset_train")
    print(f"Saved {len(all_test_prompts)} combined prompts to combined_dataset_test")
    print("All datasets built successfully.")


if __name__ == "__main__":
    main()