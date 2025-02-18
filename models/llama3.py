from typing import List

from models.base_model import BaseModel
from models.utils import llama_post_process

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class Llama3(BaseModel):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, *args, **kwargs):
        super().__init__(model, tokenizer, *args, **kwargs)
        self.device = self.model.device

    
    def post_process(self, outputs: torch.Tensor) -> str:
        output = llama_post_process(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

        return output

    
    def forward(self, inputs: dict, generation_config: dict) -> dict:
        attention_mask = torch.ones_like(inputs, dtype=torch.long, device=inputs.device)
        outputs = self.model(inputs, attention_mask=attention_mask, **generation_config)

        return outputs
    

    def generate(self, inputs: dict, generation_config: dict) -> dict:
        attention_mask = torch.ones_like(inputs, dtype=torch.long, device=inputs.device)
        outputs = self.model.generate(inputs, attention_mask=attention_mask, **generation_config)
        response = self.post_process(outputs)

        return response
    

    def get_logits(self, input: dict, generate_config: dict) -> torch.Tensor:
        outputs = self.forward(input, generate_config)
        logits = outputs.logits

        return logits
    

    def get_candidates(self, input: dict, optimize_config: dict) -> List[int]:
        generate_config = optimize_config["generate_config"]
        logits = self.get_logits(input, generate_config)[:, -1, :]

        topk = optimize_config.get("candidates_topk", 3)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(probs, topk)
        candidates = topk_tokens[0].tolist()

        return candidates
