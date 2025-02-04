from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(model, **model_params)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_params)
    
    def generate(self, input: dict) -> str:
        raise NotImplementedError("Subclass must implement this method")
    
    def get_logits(self, input: dict) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")
    
    def get_candidates(self, input: dict, position: int) -> List[str]:
        raise NotImplementedError("Subclass must implement this method")
