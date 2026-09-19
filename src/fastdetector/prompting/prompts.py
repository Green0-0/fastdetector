import json
import random
import re
from dataclasses import dataclass, field
from typing import Any


_METADATA_PLACEHOLDER = re.compile(r"<<([^<>]+)>>")


@dataclass
class Prompt:
    """A single prompt template consisting of chat turns, multiturn flag, examples, and metadata.

    Attributes:
        chat_turns: Ordered list of user-message templates. ``{{DOC}}`` is
            substituted with the sample text by :meth:`PromptSet.map`;
            ``<<COLUMN_NAME>>`` is substituted with aligned source-row
            metadata passed to :meth:`PromptSet.map`;
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

    def __init__(self, prompts: list[Prompt], train_offset: int = 0) -> None:
        """Initialize PromptSet with a list of Prompt templates.

        Args:
            prompts: List of Prompt instances to populate the training set.
            train_offset: Initial training cursor position. Values larger than
                the prompt set wrap around its length.

        Raises:
            ValueError: If ``train_offset`` is negative.
        """
        if train_offset < 0:
            raise ValueError("train_offset must be non-negative")
        self._train = list(prompts)
        self._test: list[Prompt] = []
        self._train_cursor = train_offset % len(self._train) if self._train else 0
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
        if split_index < 1:
            raise ValueError(
                f"test_fraction={test_fraction} would leave the training set "
                f"empty ({len(self._train)} prompt(s) available). Use a "
                f"smaller fraction or provide more prompts."
            )
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

    def map(
        self,
        samples: list[str],
        metadata: list[dict[str, Any]] | None = None,
        use_test: bool = False,
    ) -> tuple[list[Prompt], list[dict[str, Any]]]:
        """Map one prompt to each sample, substituting ``{{DOC}}``.

        Pulls prompts from the training or testing set via the internal cursor
        (wrapping around). For each sample, returns both the substituted
        :class:`Prompt` and a dict-of-template-fields for downstream
        bookkeeping (e.g. storing the prompt metadata in a dataset column).

        Args:
            samples: List of sample texts to map prompts onto.
            metadata: Optional source-row metadata aligned one-to-one with
                ``samples``. Each dict maps a source dataset column name to
                its value. Occurrences of ``<<COLUMN_NAME>>`` in chat turns
                are replaced with the corresponding value.
            use_test: If True, pull prompts from the test set instead of the
                train set.

        Returns:
            A tuple of:
              - A list of :class:`Prompt` objects with ``{{DOC}}`` replaced,
                one per sample.
              - A list of dicts containing the complete metadata of the
                original template prompt (before substitution).

        Raises:
            ValueError: If ``metadata`` does not contain one dict per sample.
            TypeError: If a metadata entry is not a dict.
            KeyError: If a chat turn references a metadata column that is not
                present in the corresponding metadata dict.
            ValueError: If a referenced metadata column has a null value.
        """
        if metadata is None:
            metadata = [{} for _ in samples]
        if len(metadata) != len(samples):
            raise ValueError(
                "metadata must contain exactly one dict per sample "
                f"({len(metadata)} metadata rows for {len(samples)} samples)"
            )
        if any(not isinstance(row, dict) for row in metadata):
            raise TypeError("every metadata entry must be a dict")

        templates = self.next_test(len(samples)) if use_test else self.next_train(len(samples))
        mapped: list[Prompt] = []
        prompt_labels: list[dict[str, Any]] = []
        for sample_index, (sample, sample_metadata, template) in enumerate(
            zip(samples, metadata, templates)
        ):
            requested_columns = {
                match.group(1)
                for turn in template.chat_turns
                for match in _METADATA_PLACEHOLDER.finditer(turn)
            }
            missing_columns = sorted(requested_columns - sample_metadata.keys())
            if missing_columns:
                raise KeyError(
                    f"prompt for sample {sample_index} references missing metadata "
                    f"column(s): {missing_columns}"
                )
            null_columns = sorted(
                column for column in requested_columns
                if sample_metadata[column] is None
            )
            if null_columns:
                raise ValueError(
                    f"prompt for sample {sample_index} references null metadata "
                    f"column(s): {null_columns}"
                )

            def replace_metadata(match: re.Match[str]) -> str:
                column = match.group(1)
                return str(sample_metadata[column])

            chat_turns = []
            for turn in template.chat_turns:
                mapped_turn = _METADATA_PLACEHOLDER.sub(replace_metadata, turn)
                mapped_turn = mapped_turn.replace("{{DOC}}", sample)
                chat_turns.append(mapped_turn)

            mapped.append(Prompt(
                chat_turns=chat_turns,
                use_multiturn=template.use_multiturn,
                examples=list(template.examples),
                metadata=dict(template.metadata),
            ))

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
