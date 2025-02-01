from typing import Tuple

import torch
from transformers import (
    GPTJForCausalLM,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    GPTNeoXForCausalLM,
)


def model_supported(model: str) -> Tuple[bool, str]:
    supported_models = ["gptj", "gpt2", "llama", "gemma", "gemma2", "gptneox"]
    model = model.lower().strip()

    for m in supported_models:
        if m in model:
            return True, m
    return False, None


def get_embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, Gemma2ForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
