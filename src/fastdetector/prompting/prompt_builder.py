import json
import random
import os
from typing import List
from fastdetector.prompting.prompts import Prompt

def shuffle(items: list[list[str]], seed: int = 42) -> list[list[str]]:
    """
    Copies the input list, shuffles it, and returns a new list.
    """
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    return items_copy

def resize(items: list[list[str]], target_length: int, also_shuffle: bool = True, seed: int = 42) -> list[list[str]]:
    """
    Expand a list to target_length while keeping the least number of elements duplicated.
    First, duplicates every element in the list exactly until it is just below target_length.
    Then samples the remaining elements from the original list without replacement.
    Returns a new list, without modifying the original list.

    If the number of elements desired is less than the size of the list, instead returns a subset.
    
    Maintains the original order of the list whenever possible.

    Args:
        items (list[list[str]]): List of elements to expand.
        target_length (int): Target length of the expanded list.
        also_shuffle (bool, optional): Whether to shuffle the list before sampling. Defaults to True.
        seed (int, optional): Seed for the random number generator. Defaults to 42.
    
    Returns:
        list[list[str]]: Expanded list of elements.
    """
    assert items != None and len(items) > 0, "Cannot resize an empty list."
    assert target_length >= 1, f"target_length must be >= 1, got {target_length}"
    
    rng = random.Random(seed)
    
    if target_length <= len(items):
        items_copy = list(items)
        if also_shuffle:
            rng.shuffle(items_copy)
        return items_copy[:target_length]

    result = []
    full_copies = target_length // len(items)
    remainder = target_length % len(items)
    for _ in range(full_copies):
        result.extend(items)
    if remainder > 0:
        items_copy = list(items)
        if also_shuffle:
            rng.shuffle(items_copy)
        result.extend(items_copy[:remainder])
    if also_shuffle:
        rng.shuffle(result)
    return result

def partial_stack(items_to_stack: list[list[list[str]]], min_stack_size: int = 1, max_stack_size: int = 1, seed: int = 42) -> list[list[str]]:
    """
    Takes a list of lists, where each list must have the exact same number of elements.

    Returns a list where each entry in the list is a variate length list 
    containing the i-th element from the first list up to 
    [min_stack_size, max_stack_size] lists from the lists to stack.
    
    As a result, note that some items from the list stack will be excluded.

    Returns a new list, without modifying the original list.

    Args:
        items_to_stack (list[list[list[str]]]): List of datasets to stack.
        min_stack_size (int, optional): Minimum number of lists to stack. Defaults to 1.
        max_stack_size (int, optional): Maximum number of lists to stack. Defaults to 1.
        seed (int, optional): Seed for the random number generator. Defaults to 42.
    
    Returns:
        list[list[str]]: List of sequentially combined chat turns.
    """
    assert items_to_stack != None and len(items_to_stack) > 0, "items_to_stack cannot be empty."
    assert False not in [len(lst) == len(items_to_stack[0]) for lst in items_to_stack], "All lists must have the same length."
    assert 1 <= min_stack_size <= max_stack_size, f"Invalid bounds: min ({min_stack_size}) must be <= max ({max_stack_size}) and >= 1."
    assert max_stack_size <= len(items_to_stack), f"max_stack_size ({max_stack_size}) cannot exceed the number of sets provided ({len(items_to_stack)})."

    rng = random.Random(seed)
    result = []
    for i in range(len(items_to_stack[0])):
        stack_size = rng.randint(min_stack_size, max_stack_size)
        stacked_item = []
        for j in range(stack_size):
            stacked_item.extend(items_to_stack[j][i])
        result.append(stacked_item)

    return result

def force_reformat(original: list[list[str]], only_first_message: bool = False, modified_format: str = "{{TEXT}}") -> list[list[str]]:
    """
    Takes your original list of lists, and forces a reformat to a different format. 

    Replaces {{TEXT}} with the original string present.
    
    Returns a new list, without modifying the original list.
    """
    assert "{{TEXT}}" in modified_format, "modified_format must contain the placeholder '{{TEXT}}'."

    result = []
    for chat in original:
        new_chat = []
        for i, text in enumerate(chat):
            if i == 0 or not only_first_message:
                new_chat.append(modified_format.replace("{{TEXT}}", text))
            else:
                new_chat.append(text)
        result.append(new_chat)
    return result

def apply_recursive_format(original: list[list[str]], order: str = "first", res_header: str = "{{RESP_#}}") -> list[list[str]]:
    """
    Applies recursive headers to every message in a chat sequence aside from the first.
    
    The res_header is appended to the string, either first or last depending on the order. 
    The first message in each chat sequence is left unmodified.
    
    Returns a new list, without modifying the original list.

    Args:
        original (list[list[str]]): Original list of lists.
        order (str, optional): The order to apply the format ('first' or 'last'). Defaults to "first".
        res_header (str, optional): The response header. Defaults to "{{RESP_#}}". The # symbol is replaced with the index of the last response, starting from 0.
    
    Returns:
        list[list[str]]: Formatted list of lists.
    """
    assert order in ["first", "last"], f"order must be 'first' or 'last', got {order}"
    
    result = []
    
    for chat in original:
        assert chat is not None and len(chat) > 0, "Chat lists cannot be empty."
        
        new_chat = [chat[0]]
        for i in range(1, len(chat)):
            current_res_header = res_header.replace("#", str(i - 1))
            if order == "first":
                msg = f"{current_res_header}\n{chat[i]}"
            else:
                msg = f"{chat[i]}\n{current_res_header}"
            new_chat.append(msg)

        result.append(new_chat)

    return result

def load_raw_samples(paths: list[str]) -> list[list[str]]:
    """
    Loads the samples in the specified path(s) as a single list of lists of strings.
    """    
    raw_samples = []
    for path in paths:
        assert os.path.isfile(path), f"File not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), f"JSON root in {path} must be a list."
            for i, item in enumerate(data):
                assert isinstance(item, str), f"Item at index {i} in {path} is of unsupported type: {type(item).__name__}"
                raw_samples.append([item])
    return raw_samples

def generate_dataset(prompts: list[list[str]], use_multiturn: bool = True) -> List[Prompt]:
    """
    Return a set of Prompt objects based on the input.
    """
    return [Prompt(chat_turns=chat, use_multiturn=use_multiturn) for chat in prompts]

def add_metadata(prompts: List[Prompt], key: str, value: any) -> List[Prompt]:
    """
    Adds a key-value pair to the metadata dict of each prompt in the list.
    """
    for prompt in prompts:
        prompt.metadata[key] = value
    return prompts

def add_example(prompts: List[Prompt], example: tuple[str, str]) -> List[Prompt]:
    """
    Adds a user-assistant example tuple to the examples list of each prompt in the list.
    """
    for prompt in prompts:
        prompt.examples.append(example)
    return prompts

def save_dataset(dataset: list[Prompt], name: str, path: str = "prompts/"):
    """
    Save the dataset to a JSON file.
    """
    assert dataset is not None and len(dataset) > 0, "Cannot save an empty dataset."
    
    os.makedirs(path, exist_ok=True)
    
    if not name.endswith(".json"):
        name += ".json"
        
    out_path = os.path.join(path, name)
    
    serialized_data = []
    for prompt in dataset:
        entry = {
            "chat_turns": prompt.chat_turns,
            "use_multiturn": prompt.use_multiturn,
            "examples": prompt.examples,
            "metadata": prompt.metadata
        }
        serialized_data.append(entry)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4)