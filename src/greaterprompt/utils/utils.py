import string
from typing import List, Tuple


def clean_string(prompts: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    cleaned_prompts = []

    for prompt, score in prompts:
        prompt = prompt.strip("'")
        prompt = prompt.rstrip('.')
        prompt = prompt.translate(str.maketrans('', '', string.punctuation))
        cleaned_prompts.append((prompt, score))
    
    cleaned_prompts = list(set(cleaned_prompts))
    cleaned_prompts.sort(key=lambda x: x[1])

    return cleaned_prompts
