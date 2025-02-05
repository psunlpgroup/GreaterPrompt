from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, *args, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input: dict) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")
    
    
    def generate(self, input: dict) -> str:
        raise NotImplementedError("Subclass must implement this method")
    

    def get_logits(self, input: dict) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")
    
    
    def get_candidates(self, input: dict, position: int) -> List[str]:
        raise NotImplementedError("Subclass must implement this method")
