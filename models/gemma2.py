from typing import List, Tuple

from models.base_model import BaseModel

import torch
import torch.nn.functional as F


class Gemma2(BaseModel):
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        super().__init__(model, model_params, tokenizer, tokenizer_params, *args, **kwargs)
    

    @torch.no_grad()
    def generate(self, input: dict, generate_config: dict) -> str:
        outputs = self.model.generate(**input, **generate_config)

        return outputs
    

    def get_logits(self, input: dict, generate_config: dict) -> torch.Tensor:
        generate_config = generate_config.copy()
        generate_config["max_new_tokens"] = 1
        generate_config["output_scores"] = True

        outputs = self.generate(input, generate_config)
        logits = outputs.scores[0][0]

        return logits


    def get_candidates(self, input: dict, optimize_config: dict) -> Tuple[List[str], List[float]]:
        generate_config: dict = optimize_config["generate_config"]
        logits = self.get_logits(input, generate_config)

        topk: int = optimize_config["topk"]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(probs, topk)

        return topk_tokens, topk_probs

