from typing import List

from models.base_model import BaseModel

import torch
from transformers import AutoModel, AutoTokenizer



class Llama32(BaseModel):
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        super().__init__(model, model_params, tokenizer, tokenizer_params, *args, **kwargs)
    
    def generate(self, input: dict) -> str:
        pass
    
    