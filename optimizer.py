import importlib
import logging
import time
from typing import List

from models.utils import model_supported

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    filename=f"{time.strftime('%Y%m%d_%H%M%S')}.log", filemode="w"
)


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
        logging.info('getting model predictions')
        probs = []

        with torch.enable_grad():
            generate_config = self.optimize_config["generate_config"]
            for i in range(len(y_tokens)):
                logits = self.client.get_logits(input, generate_config)[:, -1, :]
                probs.append(F.softmax(logits, dim=-1))
                next_token_id = torch.argmax(logits, dim=-1)
                next_token_id = next_token_id.unsqueeze(0)
                logging.info(f'next_token_id: {next_token_id}, next_token_id decoded: {repr(self.client.tokenizer.decode(next_token_id[0, 0]))}')
                input = torch.cat([input, next_token_id], dim=1)

        return torch.cat(probs, dim=0)
    

    def calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.client.model.zero_grad()
        loss = F.cross_entropy(y_hat, y)
        loss.backward(retain_graph=True)

        return loss
    

    def get_gradients(self, y_tokens: torch.Tensor, y_hat_probs: torch.Tensor) -> List[torch.Tensor]:
        gradients = []

        for y, y_hat in zip(y_tokens[0, :], y_hat_probs):
            logging.info(f'calculating loss')
            logging.info(f'y_token: {repr(self.client.tokenizer.decode(y))}')
            logging.info(f'y_hat_token: {repr(self.client.tokenizer.decode(torch.argmax(y_hat)))}')
            loss = self.calculate_loss(y_hat, y)
            logging.info(f'loss: {loss.item()}')
            embedding_layer = self.client.model.get_input_embeddings()
            embedding_grad = embedding_layer.weight.grad
            gradients.append(embedding_grad)

        return gradients


    def get_candidates(self, input: dict) -> List[int]:
        candidates = self.client.get_candidates(input, self.optimize_config)

        return candidates
    

    def get_reasoning(self, input: dict) -> str:
        generate_config = self.optimize_config.get("generate_config", {})
        with torch.inference_mode():
            response = self.client.generate(input, generate_config)

        return response
    
    
    def get_p_i_star(self, gradients: List[torch.Tensor], candidates: List[int]) -> int:
        logging.info('calculating p_i_star')
        p_i_star, p_i_star_grad = None, float("-inf")

        for candidate in candidates:
            token_grad = sum([torch.norm(grad[candidate], p=2) * -1 for grad in gradients])
            token_grad /= len(gradients)
            logging.info(f'candidate id: {candidate}, candidate token: {repr(self.client.tokenizer.decode(candidate))}, token_grad: {token_grad}')
            if token_grad > p_i_star_grad:
                p_i_star = candidate
                p_i_star_grad = token_grad

        return p_i_star


    def optimize(self, inputs: List[dict], extractor:str, rounds: int) -> List[List[str]]:
        outputs = []

        for i, input in enumerate(inputs):
            # TODO: why there is a space before answer?
            question, p_init, ground_truth = input["question"].strip() + " ?", " " + input["prompt"], input["answer"]
            logging.info(f'question: {question}')
            logging.info(f'p_init: {p_init}')
            logging.info(f'ground_truth: {ground_truth}\n')
            # only keep <|begin_of_text|> token for question
            question_tokens = self.client.tokenizer.encode(question, return_tensors="pt")
            question_tokens = question_tokens.to(self.client.device)
            p_tokens = self.client.tokenizer.encode(p_init, return_tensors="pt")
            p_tokens = p_tokens[:, 1:].to(self.client.device)
            p_extr_tokens = self.client.tokenizer.encode(extractor, return_tensors="pt")
            p_extr_tokens = p_extr_tokens[:, 1:].to(self.client.device)
            y_tokens = self.client.tokenizer.encode(ground_truth, return_tensors="pt")
            y_tokens = y_tokens[:, 1:].to(self.client.device)
            assert len(p_tokens[0, :]) >= 2, "Init prompt should be at least 2 words"
            p_stars = []
            idx = 1

            for j in tqdm(range(rounds), desc=f"Optimizing {i} / {len(inputs)}"):
                torch.cuda.empty_cache()
                # calculate p_i, if it is the first token, skip
                logging.info(f'Round {j}, p_idx: {idx}')
                logging.info(f'p_tokens: {p_tokens}')
                logging.info(f'p_tokens decoded: {repr(self.client.tokenizer.decode(p_tokens[0, :]))}')

                # get candidates for p_i by using x + p_0 ... p_i-1
                token_i = p_tokens[:, idx]
                logging.info(f'token_i: {token_i}, token_i decoded: {repr(self.client.tokenizer.decode(token_i))}')
                input_ids = torch.cat([question_tokens, p_tokens[:, :idx]], dim=1)
                logging.info(f'input text for candidate generation: {repr(self.client.tokenizer.decode(input_ids[0, :]))}')
                candidates = self.get_candidates(input_ids)
                candidates.append(int(token_i[0]))
                logging.info(f'candidates: {candidates}')
                logging.info(f'candidates decoded: {[repr(self.client.tokenizer.decode(c)) for c in candidates]}')

                # get reasoning chain r by x + p
                input_ids = torch.cat([question_tokens, p_tokens], dim=1)
                reasoning_chain = self.get_reasoning(input_ids)
                logging.info(f'reasoning_chain:\n {reasoning_chain}')
                r_tokens = self.client.tokenizer.encode(reasoning_chain, return_tensors="pt")
                r_tokens = r_tokens.to(self.client.device)

                # use x + p + r + p_extractor to get logits of y_hat
                input_ids = torch.cat([question_tokens, p_tokens, r_tokens, p_extr_tokens], dim=1)
                y_hat_probs = self.get_pred_probs(input_ids, y_tokens)
                gradients = self.get_gradients(y_tokens, y_hat_probs)

                # calculate gradient for each candidate to get p_i_star
                p_i_star = self.get_p_i_star(gradients, candidates)
                logging.info(f'p_i_star: {p_i_star}, p_i_star decoded: {repr(self.client.tokenizer.decode(p_i_star))}')
                p_tokens[:, idx] = p_i_star
                logging.info(f'p_tokens after updating: {p_tokens}')
                logging.info(f'p_tokens decoded: {repr(self.client.tokenizer.decode(p_tokens[0, :]))}')

                # if p_i_star is a period, truncate the prompt and star from the beginning
                p_i_star_token = self.client.tokenizer.decode(p_i_star)
                if p_i_star_token.strip() == ".":
                    p_tokens = p_tokens[:, :idx + 1]
                    p_stars.append(self.client.tokenizer.decode(p_tokens[0, :idx + 1], skip_special_tokens=True))
                    idx = 1
                    logging.info(f'p_i_star is a period, truncate the prompt and star from the beginning')
                # elif p_i_star is not a period and it's the last token, append a dummy
                # token for the next round of candidate generation
                elif p_i_star_token.strip() != "." and idx == len(p_tokens[0, :]) - 1:
                    dummy_token = torch.tensor([[0]], device=p_tokens.device)
                    p_tokens = torch.cat([p_tokens, dummy_token], dim=1)
                    idx += 1
                    logging.info(f'p_i_star is not a period and it is the last token, extend the prompt')
                else:
                    idx = (idx + 1) % len(p_tokens[0, :])
                    logging.info(f'p_i_star is not a period and it is not the last token, update the index')
                logging.info(f'\n')

            outputs.append(p_stars if p_stars else [self.client.tokenizer.decode(p_tokens[0, :], skip_special_tokens=True)])

        return outputs
