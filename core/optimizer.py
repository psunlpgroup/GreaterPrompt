import importlib

from models.utils import model_supported

import torch
from typing import List


class GreaterOptimizer:
    def __init__(
        self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict,
        optimize_config: dict, *args, **kwargs
    ):
        self.optimize_config = optimize_config
        self._init_agents(model, model_params, tokenizer, tokenizer_params)
    

    def _init_agents(self, model: str, model_params: dict, tokenizer: str, tokenizer_params: dict):
        supported, model_name = model_supported(model)
        assert supported, f"Model: {model} is not supported"

        model_fn = getattr(importlib.import_module("models"), model_name)
        self.model = model_fn(model, model_params, tokenizer, tokenizer_params)
    

    @torch.no_grad()
    def _get_logits(self, input: dict) -> torch.Tensor:
        topk = self.optimize_config["topk"]


    def _get_candidates(self, input: dict, position: int) -> List[str]:
        pass


    def _generate_p_star(self, input: List[dict]) -> List[str]:
        pass


    def optimize(self, inputs: List[dict], rounds: int) -> List[str]:
        p_stars: List[str] = []
        step = self.optimize_config["intersection_num"] if self.optimize_config["intersection"] else 1

        for i in range(0, len(inputs), step):
            input_batch = inputs[i:i+step]
            for _ in range(rounds):
                pass

        return p_stars
