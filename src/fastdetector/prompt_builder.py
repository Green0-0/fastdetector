from fastdetector.prompts import Prompt
from typing import List

def resize(items: list, target_length: int, also_shuffle: bool = True, seed: int = 42) -> list:
    """
    Expand a list to target_length while keeping the least number of elements duplicated.
    First, duplicates every element in the list exactly until it is just below target_length.
    Then samples the remaining elements from the original list without replacement.
    Returns a new list, without modifying the original list.

    If the number of elements desired is less than the size of the list, instead returns a subset.
    
    Maintains the original order of the list whenever possible.

    Args:
        items (list): List of elements to expand.
        target_length (int): Target length of the expanded list.
        also_shuffle (bool, optional): Whether to shuffle the list before sampling. Defaults to True.
        seed (int, optional): Seed for the random number generator. Defaults to 42.
    
    Returns:
        list: Expanded list of elements.
    """
    pass

def partial_stack(items_to_stack:list[list], max_stack_size, min_stack_size = 1):
    """
    Takes a list of lists, where each list must have the exact same number of elements.

    Returns a list where each entry in the list is a variate length list 
    containing the i-th element from the first list up to 
    [min_stack_size, max_stack_size] lists from the lists to stack.
    
    As a result, note that some items from the list stack will be excluded.

    Returns a new list, without modifying the original list.

    Args:
        items_to_stack (list[list]): List of lists to stack.
        max_stack_size (int): Maximum number of lists to stack.
        min_stack_size (int, optional): Minimum number of lists to stack. Defaults to 1.
    
    Returns:
        list[list]: List of lists.
    """
    pass

def force_reformat(original: list[list[str]], only_first_message = False, modified_format="{{TEXT}}") -> list[list[str]]:
    """
    Takes your original list of lists, and forces a reformat to a different format. 

    Replaces {{TEXT}} with the original string present.
    
    Returns a new list, without modifying the original list.

    Args:
        original (list[list[str]]): Original list of lists.
        only_first_message (bool, optional): Whether to only reformat the first message. Defaults to False.
        modified_format (str, optional): The format to reformat to. Defaults to "{{TEXT}}".
    
    Returns:
        list[list[str]]: Reformated list of lists.
    """
    pass

def apply_multiturn_format(original: list[list[str]], format_type="recursive", order="first", doc_header="{{DOC}}", res_header="{{RESP_#}}") -> list[list[str]]:
    """
    Apply a multiturn/recursive format to the original list of lists.
    
    This is done by appending either the doc_header or res_header to the string, either first or last depending on the order. Doc_header is always used for the first turn, and res_header is always used for subsequent turns. Format_type = "recursive" implies no multiturn is used, "multiturn" implies a singular chat, "multiturn_recursive" implies a multiturn where the previous response header is sent again at the start of the next turn.

    Returns a new list, without modifying the original list.

    Args:
        original (list[list[str]]): Original list of lists.
        format_type (str, optional): The format to apply. Defaults to "recursive".
        order (str, optional): The order to apply the format. Defaults to "first".
        doc_header (str, optional): The doc header. Defaults to "{{DOC}}".
        res_header (str, optional): The response header. Defaults to "{{RESP_#}}". The # symbol is replaced with the index of the last response, starting from 0.
    
    Returns:
        list[list[str]]: Formatted list of lists.
    """
    pass

def load_raw_samples(paths: list[str]) -> list[list[str]]:
    """
    Loads all the raw samples from the "sample_prompts" directory as a single list of lists of strings.
    
    Args:
        paths (list[str]): List of paths to load samples from.
    
    Returns:
        list[list[str]]: List of lists of strings.
    """
    pass

def generate_dataset(prompts: list[list[str]]) -> List[Prompt]:
    """
    Return a set of Prompt objects based on the input.

    Args:
        prompts (list[list[str]]): List of prompts to use.
        
    Returns:
        list[Prompt]: List of Prompt objects.
    """
    pass

def save_dataset(dataset: list[Prompt], name: str, path: str = "prompts/"):
    """
    Save the dataset to a JSON file.

    Args:
        dataset (list[Prompt]): Dataset to save.
        name (str): Name of the file to save the dataset to.
        path (str): Path to save the dataset to.
    """
    pass

