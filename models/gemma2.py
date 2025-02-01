from models.base_model import BaseModel

import torch


class Gemma2(BaseModel):
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        super().__init__(model, model_params, tokenizer, tokenizer_params, *args, **kwargs)
    
    def generate(self, input: dict) -> str:
        pass
    
    def get_logits(self, input: dict) -> torch.Tensor:
        pass
