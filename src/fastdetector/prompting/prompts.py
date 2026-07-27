import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Prompt:
    """A single prompt template consisting of chat turns, multiturn flag, examples, and metadata.

    Attributes:
        chat_turns: Ordered list of user-message templates. ``{{DOC}}`` is
            substituted with the sample text by :meth:`PromptSet.map`;
            ``{{RESP_N}}`` is substituted with the model response from turn
            N by :mod:`fastdetector.generator`.
        use_multiturn: If True, all turns are sent as a single multi-turn
            conversation. If False, only the last turn is sent (the earlier
            turns are used only for ``{{RESP_N}}`` substitution context).
        examples: Few-shot examples as ``(user, assistant)`` string pairs.
        metadata: Free-form metadata dict (e.g. ``{"PROMPT_TYPE": ...}``).
    """

    chat_turns: list[str]
    use_multiturn: bool
    examples: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PromptSet:
    """A cursor-based iterator over a list of :class:`Prompt` objects.

    Holds separate train/test lists with independent cursors. The
    ``next_train`` / ``next_test`` methods return prompts in order, wrapping
    around when the cursor reaches the end of the list.
    """

    def __init__(self, prompts: list[Prompt]) -> None:
        """Initialize PromptSet with a list of Prompt templates.

        Args:
            prompts: List of Prompt instances to populate the training set.
        """
        self._train = list(prompts)
        self._test: list[Prompt] = []
        self._train_cursor = 0
        self._test_cursor = 0

    def generate_test_split(self, test_fraction: float) -> None:
        """Partition the training set into a new testing set (in-place).

        Moves a fraction of the training prompts into the test set. If a test
        set already exists, it is merged back into the training set before
        re-splitting.

        Args:
            test_fraction: Fraction of the training set to move into the test
                set. Must be strictly between 0 and 1.

        Raises:
            ValueError: if ``test_fraction`` is not in (0, 1).
        """
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1 (exclusive).")

        if self._test and len(self._test) > 0:
            print("There is already a test set defined. Merging it with the training set before re-splitting.")
            self.clear_test_set()

        split_index = int(len(self._train) * (1.0 - test_fraction))
        self._test = self._train[split_index:]
        self._train = self._train[:split_index]

    def clear_test_set(self) -> None:
        """Move all test-set prompts back into the training set and reset cursors.

        Returns:
            None.
        """
        if not self._test or len(self._test) == 0:
            print("There is no test set to clear.")
            return
        self._train.extend(self._test)
        self._test = []
        self._train_cursor = 0
        self._test_cursor = 0

    def map(self, samples: list[str], use_test: bool = False) -> tuple[list[Prompt], list[dict[str, Any]]]:
        """Map one prompt to each sample, substituting ``{{DOC}}``.

        Pulls prompts from the training or testing set via the internal cursor
        (wrapping around). For each sample, returns both the substituted
        :class:`Prompt` and a dict-of-template-fields for downstream
        bookkeeping (e.g. storing the prompt metadata in a dataset column).

        Args:
            samples: List of sample texts to map prompts onto.
            use_test: If True, pull prompts from the test set instead of the
                train set.

        Returns:
            A tuple of:
              - A list of :class:`Prompt` objects with ``{{DOC}}`` replaced,
                one per sample.
              - A list of dicts containing the complete metadata of the
                original template prompt (before substitution).
        """
        templates = self.next_test(len(samples)) if use_test else self.next_train(len(samples))
        mapped: list[Prompt] = []
        prompt_labels: list[dict[str, Any]] = []
        for sample, template in zip(samples, templates):
            mapped.append(Prompt(
                chat_turns=[turn.replace("{{DOC}}", sample) for turn in template.chat_turns],
                use_multiturn=template.use_multiturn,
                examples=list(template.examples),
                metadata=dict(template.metadata),
            ))

            # NOTE: FOR SOME REASON HF DOESN'T ALLOW EMPTY DICTIONARIES...
            meta = dict(template.metadata)
            if not meta:
                meta["_dummy"] = True

            prompt_labels.append({
                "chat_turns": template.chat_turns,
                "use_multiturn": template.use_multiturn,
                "examples": template.examples,
                "metadata": meta,
            })
        return mapped, prompt_labels

    def next_train(self, num: int) -> list[Prompt]:
        """Return ``num`` prompts from the training set, advancing the cursor.

        Wraps around to the start of the list when the cursor reaches the end.

        Args:
            num: Number of prompts to return.

        Returns:
            A list of ``num`` prompts.

        Raises:
            RuntimeError: if the training set is empty.
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
        """Return ``num`` prompts from the testing set, advancing the cursor.

        Wraps around to the start of the list when the cursor reaches the end.

        Args:
            num: Number of prompts to return.

        Returns:
            A list of ``num`` prompts.

        Raises:
            RuntimeError: if the test set is empty. Use :meth:`generate_test_split`
                to create one.
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
        """Return all prompts currently in the training set (without advancing the cursor).

        Returns:
            List of training Prompt objects.
        """
        return list(self._train)

    def get_test(self) -> list[Prompt]:
        """Return all prompts currently in the testing set (without advancing the cursor).

        Returns:
            List of testing Prompt objects.
        """
        return list(self._test)

    def reset(self) -> None:
        """Reset both the training and testing cursors to 0.

        Returns:
            None.
        """
        self._train_cursor = 0
        self._test_cursor = 0

    def shuffle(self, seed: int) -> None:
        """Shuffle the training prompts in-place using the given seed.

        Resets the training cursor to 0 after shuffling. The test set is not
        affected.

        Args:
            seed: PRNG seed for reproducibility.

        Returns:
            None.
        """
        rng = random.Random(seed)
        rng.shuffle(self._train)
        self._train_cursor = 0


def load_prompts(all_paths: list[str]) -> list[Prompt]:
    """Load prompts from JSON files.

    Each file must contain a JSON list of objects with ``chat_turns``
    (``list[str]``), ``use_multiturn`` (``bool``), ``examples``
    (``list[tuple[str, str]]``), and ``metadata`` (``dict``) fields.

    Args:
        all_paths: List of JSON file paths to load prompts from.

    Returns:
        A list of :class:`Prompt` objects, concatenated in the order the paths
        are given.

    Raises:
        ValueError: if a path doesn't end with ``.json``, the file content is
            not a JSON list, or an entry is missing a required key.
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
            if "chat_turns" not in entry or "use_multiturn" not in entry or "examples" not in entry or "metadata" not in entry:
                raise ValueError(f"Entry {i} in {path} must have 'chat_turns', 'use_multiturn', 'examples', and 'metadata' keys.")

            prompts.append(Prompt(
                chat_turns=entry["chat_turns"],
                use_multiturn=entry["use_multiturn"],
                examples=entry["examples"],
                metadata=entry["metadata"],
            ))

    return prompts