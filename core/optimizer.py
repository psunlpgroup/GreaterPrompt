import importlib
from typing import List, Tuple

from models.utils import model_supported

import torch
from torch.nn import functional as F


P_EXTRACTOR = "Only return the exact answer. Therefore, the final answer (use exact format: '$ True' or '$ False') is $ "


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


    def get_pred_probs(self, input: dict) -> torch.Tensor:
        with torch.enable_grad():
            generate_config = self.optimize_config["generate_config"]
            logits = self.client.get_logits(input, generate_config)
            probs = F.softmax(logits, dim=-1)

        return probs
    

    def calculate_loss(self, y_tokens: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(probs, y_tokens)
        loss.backward(retain_graph=True)

        return loss


    def get_candidates(self, input: dict) -> Tuple[List[str], List[float]]:
        candidates = self.client.get_candidates(input, self.optimize_config)

        return candidates
    

    def get_reasoning(self, input: dict) -> str:
        generate_config = self.optimize_config.get("reasoning_config", {})
        with torch.inference_mode():
            response = self.client.generate(input, generate_config)

        return self.client.post_process(response)
    
    
    def get_p_i_start(self, candidates: List[str]) -> str:
        embedding_layer = self.client.model.get_input_embeddings()
        embedding_grad = embedding_layer.weight.grad

        p_i_start, p_i_start_grad = None, float("inf")

        for token in candidates:
            token_id = self.client.tokenizer.encode(token, return_tensors="pt")
            token_grad = torch.norm(embedding_grad[token_id], p=2)
            if token_grad < p_i_start_grad:
                p_i_start = token
                p_i_start_grad = token_grad

        return p_i_start


    def optimize(self, inputs: List[dict], rounds: int) -> Tuple[List[str], List[dict]]:
        outputs, meta_info = [], []

        for input in inputs:
            # TODO: hard code here, model outpus start with a space
            question, p_init, ground_truth = input["question"], input["prompt"], " " + input["answer"]
            p_tokens = self.client.tokenizer.tokenize(p_init)
            y_tokens = self.client.tokenizer.encode(ground_truth, return_tensors="pt")
            y_tokens = y_tokens[0, 1:].to(self.client.device)
            assert len(p_tokens) >= 2, "Init prompt should be at least 2 words"

            for i in range(rounds):
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

                y_hat_probs = self.get_pred_probs(input_ids)
                loss = self.calculate_loss(y_tokens, y_hat_probs)

                # calculate gradient for each candidate to get p_i_start
                p_i_start = self.get_p_i_start(candidates)
                p_tokens[idx] = p_i_start

            outputs.append("".join(p_tokens))
            meta_info.append({
                "question": question,
                "p_init": p_init,
                "p_star": "".join(p_tokens)}
            )

        return outputs, meta_info
    

    def batch_optimize(self, inputs: List[dict], rounds: int) -> List[str]:
        p_stars: List[str] = []

        for input in inputs:
            pass

        return p_stars
