from typing import Tuple

import torch
from transformers import (
    GPTJForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM, 
    LlamaForCausalLM, GemmaForCausalLM, Gemma2ForCausalLM
)


# TODO: hard code here
def model_supported(model: str) -> Tuple[bool, str]:
    supported_models = {
        "Gemma-2": "Gemma2", 
        "Llama-3.1": "Llama31"
    }

    for k, v in supported_models.items():
        if k.lower() in model.lower():
            return True, v
    return False, None


def get_embedding_layer(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    elif isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, Gemma2ForCausalLM):
        return model.model.embed_tokens
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


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


def get_embeddings(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GemmaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, Gemma2ForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def llama_post_process(text: str) -> str:
    text = text.split("<|end_header_id|>")[-1]
    text = text.replace("<|eot_id|>","")

    return text
