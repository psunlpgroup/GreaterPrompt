import importlib
from typing import List, Tuple

from models.utils import llama_post_process, model_supported

import torch


P_EXTRACTOR = "Only return the exact answer. Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "
T = 105


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

        model_class = getattr(importlib.import_module("models"), model_name)
        self.client = model_class(model, model_params, tokenizer, tokenizer_params)


    def get_prediction(self, input: dict, logits_size: int) -> str:
        logits = []

        with torch.enable_grad():
            for _ in range(logits_size):
                logit = self.client.get_logits(input, self.optimize_config["generate_config"])
                logits.append(logit.detach().cpu())

        return logits


    def get_candidates(self, input: dict) -> Tuple[List[str], List[float]]:
        candidates = self.client.get_candidates(input, self.optimize_config)

        return candidates
    

    def get_reasoning(self, input: dict) -> str:
        generate_config = self.optimize_config.get("reasoning_config", {})
        with torch.inference_mode():
            response = self.client.generate(input, generate_config)

        return self.client.post_process(response)
    

    def optimize(self, inputs: List[dict], rounds: int) -> List[str]:
        outputs: List[str] = []

        for input in inputs:
            question, p_init, ground_truth = input["question"], input["prompt"], input["answer"]
            p_tokens = self.client.tokenizer.tokenize(p_init)
            y_token = self.client.tokenizer.tokenize(ground_truth)
            assert len(p_tokens) >= 2, "Init prompt should be at least 2 words"

            for i in range(T):
                # calculate p_i, if i == 0, skip
                idx = i % len(p_tokens)
                if idx == 0: continue

                # get candidates for p_i using x + p_0 ... p_i-1
                token = p_tokens[idx]
                input_text = f'{question} {"".join(p_tokens[:idx])}'
                input_ids = self.client.tokenizer.encode(input_text, return_tensors="pt")

                candidates = self.get_candidates(input_ids)
                candidates += [token]

                # get reasoning chain r by x + p
                reasoning_chain = self.get_reasoning(input_ids)
                input_text = f'{question} {"".join(p_tokens)} {reasoning_chain}'
                input_ids = self.client.tokenizer.encode(input_text, return_tensors="pt")

                # then using x + r + p to get raw response and extract y_hat
                input_text = f'{question} {"".join(p_tokens)} {reasoning_chain} {P_EXTRACTOR}'
                input_ids = self.client.tokenizer.encode(input_text, return_tensors="pt")

                y_hat = self.get_prediction(input_ids)
                


        return outputs
    

    def batch_optimize(self, inputs: List[dict], rounds: int) -> List[str]:
        p_stars: List[str] = []

        for input in inputs:
            pass

        return p_stars
