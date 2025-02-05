import importlib
from typing import List, Tuple

from models.utils import model_supported

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class GreaterOptimizer:
    def __init__(
            self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, optimize_config: dict, *args, **kwargs
        ):
        self.optimize_config = optimize_config
        self._init_agents(model, tokenizer)
    

    def _init_agents(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        supported, model_name = model_supported(model)
        assert supported, f"Model: {model} is not supported"

        model_class = getattr(importlib.import_module("models"), model_name)
        self.client = model_class(model, tokenizer)


    def get_pred_probs(self, input: dict, y_tokens: torch.Tensor) -> torch.Tensor:
        probs = []

        with torch.enable_grad():
            generate_config = self.optimize_config["generate_config"]
            for i in range(-len(y_tokens), 0, 1):
                logits = self.client.get_logits(input, generate_config)[:, i, :]
                probs.append(F.softmax(logits, dim=-1))
        
        resp = ""
        for p in probs:
            resp += self.client.tokenizer.decode(p.argmax(dim=-1))

        return torch.cat(probs, dim=0)
    

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.client.model.zero_grad()
        loss = F.cross_entropy(y_hat, y)
        loss.backward(retain_graph=True)

        return loss
    

    def get_gradients(self, y_tokens: torch.Tensor, probs: torch.Tensor) -> List[torch.Tensor]:
        gradients = []

        for y, y_hat in zip(y_tokens, probs):
            loss = self.calculate_loss(y_hat, y)
            embedding_layer = self.client.model.get_input_embeddings()
            embedding_grad = embedding_layer.weight.grad
            gradients.append(embedding_grad)

        return gradients


    def get_candidates(self, input: dict) -> List[str]:
        candidates = self.client.get_candidates(input, self.optimize_config)

        return candidates
    

    def get_reasoning(self, input: dict) -> str:
        generate_config = self.optimize_config.get("reasoning_config", {})
        with torch.inference_mode():
            response = self.client.generate(input, generate_config)

        return response
    
    
    def get_p_i_star(self, gradients: List[torch.Tensor], candidates: List[str]) -> str:
        p_i_star, p_i_star_grad = None, float("-inf")

        for token in candidates:
            token_id = self.client.tokenizer.encode(token, return_tensors="pt")
            token_grad = sum([torch.norm(grad[token_id], p=2) * -1 for grad in gradients])
            token_grad /= len(gradients)
            if token_grad > p_i_star_grad:
                p_i_star = token
                p_i_star_grad = token_grad

        return p_i_star


    def optimize(self, inputs: List[dict], extractor:str, rounds: int) -> Tuple[List[List[str]], List[dict]]:
        outputs, meta_info = [], []

        for i, input in enumerate(inputs):
            question, p_init, ground_truth = input["question"], input["prompt"], " " + input["answer"]
            p_tokens = self.client.tokenizer.tokenize(p_init)
            y_tokens = self.client.tokenizer.encode(ground_truth, return_tensors="pt")
            # TODO: hard code here, llama tokens start with a <s>
            y_tokens = y_tokens[0, 1:].to(self.client.device)
            assert len(p_tokens) >= 2, "Init prompt should be at least 2 words"
            p_stars = []

            for i in tqdm(range(rounds), desc=f"Optimizing {i} / {len(inputs)}"):                
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
                input_text = f'{question} {"".join(p_tokens)} {reasoning_chain} {extractor}'
                input_ids = self.client.tokenizer.encode(input_text, return_tensors="pt")

                # get logits of y_hat and use backward propagation to get gradients
                y_hat_probs = self.get_pred_probs(input_ids, y_tokens)
                gradients = self.get_gradients(y_tokens, y_hat_probs)

                # calculate gradient for each candidate to get p_i_star
                p_i_star = self.get_p_i_star(gradients, candidates)
                p_tokens[idx] = p_i_star

                # if p_i_star is a period, truncate the prompt and star from the beginning
                if p_i_star.strip() == ".":
                    p_tokens = p_tokens[:idx + 1]
                    p_stars.append("".join(p_tokens).strip())
                    idx = 1
                # elif p_i_star is not a period and it's the last token, append a dummy token
                # for next candidate generation
                elif p_i_star.strip() != "." and idx == len(p_tokens) - 1:
                    p_tokens.append("#")
                    idx += 1
                else:
                    idx += 1

            outputs.append(p_stars if p_stars else ["".join(p_tokens).strip()])
            meta_info.append(input)

        return outputs, meta_info
    

    def batch_optimize(self, inputs: List[dict], rounds: int) -> List[str]:
        p_stars: List[str] = []

        for input in inputs:
            pass

        return p_stars
