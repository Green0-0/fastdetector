from operator import mod
from fastdetector.prompt_builder import shuffle, resize, partial_stack, force_reformat, apply_recursive_format, load_raw_samples, generate_dataset, save_dataset

def main():
    samples = load_raw_samples(["sample_prompts/audience.json", "sample_prompts/clarify.json", "sample_prompts/edit.json", "sample_prompts/elaboration.json", "sample_prompts/restructure.json", "sample_prompts/tone.json"])
    samples = partial_stack([shuffle(samples, i) for i in range(0, 3)], 3)
    samples = force_reformat(samples, only_first_message=True, modified_format="{{DOC}}\n{{TEXT}}")
    samples = force_reformat(samples, only_first_message=True, modified_format="{{TEXT}}\nOutput the full new text with no extra statements or commentations.")
    prompts = generate_dataset(samples, use_multiturn=True)
    save_dataset(prompts, "testing_dataset")

if __name__ == "__main__":
    main()