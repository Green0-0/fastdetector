from fastdetector.prompt_builder import shuffle, resize, partial_stack, force_reformat, apply_recursive_format, load_raw_samples, generate_dataset, save_dataset, add_metadata

def main():
    print("Collecting prompts...")
    samples = load_raw_samples(["sample_prompts/revise/audience.json", "sample_prompts/revise/clarify.json", "sample_prompts/revise/edit.json", "sample_prompts/revise/elaboration.json", "sample_prompts/revise/restructure.json", "sample_prompts/revise/tone.json"])
    print(f"Loaded {len(samples)} prompts.")
    
    print("Stacking prompts...")
    samples = partial_stack([shuffle(samples, i) for i in range(0, 3)], 1, 3)
    print(f"Stacked {len(samples)} prompts.")

    print("Formatting prompts...")
    samples = force_reformat(samples, only_first_message=True, modified_format="{{DOC}}\n{{TEXT}}")
    
    print("Building multiturn variant...")
    mt_samples = force_reformat(samples, only_first_message=False, modified_format="{{TEXT}}\nOutput the full new text with no extra statements or commentations.")
    mt_prompts = generate_dataset(mt_samples, use_multiturn=True)
    add_metadata(mt_prompts, "PROMPT_TYPE", "revise")
    add_metadata(mt_prompts, "VERSION", "multiturn")
    print(f"Generated {len(mt_prompts)} prompts.")
    print("Saving prompts...")
    save_dataset(mt_prompts, "testing_multiturn_dataset")
    print("Saved prompts to testing_multiturn_dataset.")

    print("Building recursive variant...")
    samples = force_reformat(samples, only_first_message=False, modified_format="{{TEXT}}\nOutput the full new text with no extra statements or commentations.")
    r_samples = apply_recursive_format(samples)
    r_prompts = generate_dataset(r_samples, use_multiturn=False)
    add_metadata(r_prompts, "PROMPT_TYPE", "revise")
    add_metadata(r_prompts, "VERSION", "recursive")
    print(f"Generated {len(r_prompts)} prompts.")
    print("Saving prompts...")
    save_dataset(r_prompts, "testing_recursive_dataset")
    print("Saved prompts to testing_recursive_dataset.")

if __name__ == "__main__":
    main()