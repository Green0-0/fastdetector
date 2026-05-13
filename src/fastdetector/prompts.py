import json
import random
from dataclasses import dataclass

@dataclass
class Prompt:
    """A single prompt entry consisting of an ordered sequence of chat turns and a multiturn flag."""
    chat_turns: list[str]
    use_multiturn: bool

class PromptSet:
    def __init__(self, prompts: list[Prompt]):
        self._train = list(prompts)
        self._test: list[Prompt] = []
        self._train_cursor = 0
        self._test_cursor = 0

    def generate_test_split(self, test_fraction: float):
        """
        Internally partitions the training set into a new testing set (not duplicated from the training set). Must be called before utilizing a testing set.

        Args:
            test_fraction (float): Fraction of the training set to be used as the testing set.
        """
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1 (exclusive).")

        if self._test and len(self._test) > 0:
            print("There is already a test set defined. Merging it with the training set before re-splitting.")
            self.clear_test_set()

        split_index = int(len(self._train) * (1.0 - test_fraction))
        self._test = self._train[split_index:]
        self._train = self._train[:split_index]

    def clear_test_set(self):
        """
        Clears the test set, adding the prompts back into the training set.
        """
        if not self._test or len(self._test) == 0:
            print("There is no test set to clear.")
            return
        self._train.extend(self._test)
        self._test = []
        self._train_cursor = 0
        self._test_cursor = 0

    def map(self, samples: list[str], use_test: bool = False) -> tuple[list[Prompt], list[str]]:
        """
        Maps one prompt to each of the samples in the list. Pulls prompts
        from the training or testing set via the internal cursor (wrapping around).
        Replaces {{DOC}} in all chat turns with the corresponding sample text.

        Args:
            samples: List of sample texts to map prompts onto.
            use_test: If True, pull prompts from the test set instead of the train set.

        Returns:
            A tuple of:
              - A list of Prompt objects with {{DOC}} replaced, one per sample.
              - A list of JSON-serialized original template chat_turns (before substitution).
        """
        templates = self.next_test(len(samples)) if use_test else self.next_train(len(samples))
        mapped: list[Prompt] = []
        prompt_labels: list[str] = []
        for sample, template in zip(samples, templates):
            mapped.append(Prompt(
                chat_turns=[turn.replace("{{DOC}}", sample) for turn in template.chat_turns],
                use_multiturn=template.use_multiturn,
            ))
            prompt_labels.append(json.dumps(template.chat_turns))
        return mapped, prompt_labels

    def next_train(self, num: int) -> list[Prompt]:
        """
        Returns `num` prompts from the training set, advancing an internal cursor. 
        """
        if not self._train:
            raise RuntimeError("The training set is empty.")

        result = []
        for _ in range(num):
            self._train_cursor %= len(self._train)
            result.append(self._train[self._train_cursor])
            self._train_cursor += 1
        return result
    
    def next_test(self, num: int) -> list[Prompt]:
        """
        Returns `num` prompts from the testing set, advancing an internal cursor. 
        """
        if not self._test:
            raise RuntimeError("The testing set is empty. Use generate_test_split() to create one.")

        result = []
        for _ in range(num):
            self._test_cursor %= len(self._test)
            result.append(self._test[self._test_cursor])
            self._test_cursor += 1
        return result

    def get_train(self) -> list[Prompt]:
        """
        Returns all of the prompts currently in the training set, without touching the internal train set cursor.
        """
        return list(self._train)
    
    def get_test(self) -> list[Prompt]:
        """
        Returns all of the prompts currently in the testing set, without touching the internal test set cursor.
        """
        return list(self._test)
    
    def reset(self):
        """
        Resets the internal cursors for both the training and testing sets.
        """
        self._train_cursor = 0
        self._test_cursor = 0

    def shuffle(self, seed: int):
        """
        Shuffles the training prompts only using a given seed.
        """
        rng = random.Random(seed)
        rng.shuffle(self._train)
        self._train_cursor = 0


def load_prompts(all_paths: list[str]) -> PromptSet:
    """
    Load prompts from JSON files. Each file must contain a JSON list of objects
    with "chat_turns" (list[str]) and "use_multiturn" (bool) fields.

    Args:
        all_paths (list[str]): List of JSON file paths to load prompts from.

    Returns:
        PromptSet: Set of prompts.
    """
    prompts: list[Prompt] = []

    for path in all_paths:
        if not path.endswith(".json"):
            raise ValueError(f"Only JSON files are supported, got: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")

        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise ValueError(f"Entry {i} in {path} must be an object, got {type(entry).__name__}.")
            if "chat_turns" not in entry or "use_multiturn" not in entry:
                raise ValueError(f"Entry {i} in {path} must have 'chat_turns' and 'use_multiturn' keys.")

            prompts.append(Prompt(
                chat_turns=entry["chat_turns"],
                use_multiturn=entry["use_multiturn"],
            ))

    return PromptSet(prompts)