from typing import List

from models.base_model import BaseModel
from models.utils import llama_post_process

import torch
from torch.nn import functional as F


class Llama31(BaseModel):
    def __init__(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict, *args, **kwargs):
        super().__init__(model, model_params, tokenizer, tokenizer_params, *args, **kwargs)
        self.device = self.model.device
        self.end_token = self.tokenizer.eos_token

    
    def post_process(self, outputs: torch.Tensor) -> str:
        output = llama_post_process(self.tokenizer.decode(outputs[0]))
        
        return output

    
    def forward(self, inputs: dict, generation_config: dict) -> dict:
        inputs = inputs.to(self.device)
        attention_mask = torch.ones_like(inputs).to(self.device)
        outputs = self.model(inputs, attention_mask=attention_mask, **generation_config)

        return outputs
    

    def generate(self, inputs: dict, generation_config: dict) -> dict:
        inputs = inputs.to(self.device)
        attention_mask = torch.ones_like(inputs).to(self.device)
        outputs = self.model.generate(inputs, attention_mask=attention_mask, **generation_config)

        return outputs
    

    def get_logits(self, input: dict, generate_config: dict) -> torch.Tensor:
        outputs = self.forward(input, generate_config)
        logits = outputs.logits[:, -1, :]

        return logits
    

    def get_candidates(self, input: dict, optimize_config: dict) -> List[str]:
        generate_config = optimize_config["generate_config"]
        logits = self.get_logits(input, generate_config)

        topk = optimize_config.get("candidates_topk", 3)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(probs, topk)
        topk_tokens = topk_tokens.cpu().numpy()[0]

        candidates = [self.tokenizer.decode(token) for token in topk_tokens]

        return candidates
