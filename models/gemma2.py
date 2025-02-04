from typing import List, Tuple

from models.base_model import BaseModel

import torch
import torch.nn.functional as F


class Gemma2(BaseModel):
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        super().__init__(model, model_params, tokenizer, tokenizer_params, *args, **kwargs)
        self.device = self.model.device
    

    def generate(self, input: dict, generate_config: dict) -> str:
        input = input.to(self.device)
        outputs = self.model.generate(**input, **generate_config)

        return outputs
    

    def get_logits(self, input: dict, generate_config: dict) -> torch.Tensor:
        outputs = self.generate(input, generate_config)
        logits = outputs.scores

        return logits


    def get_candidates(self, input: dict, optimize_config: dict) -> Tuple[List[str], List[float]]:
        generate_config: dict = optimize_config["generate_config"]
        logits = self.get_logits(input, generate_config)

        topk: int = optimize_config["candidates_topk"]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(probs, topk)

        candidates = [self.tokenizer.decode(token) for token in topk_tokens[0]]

        return candidates, topk_probs
