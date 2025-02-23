from typing import Tuple

from transformers import AutoModelForCausalLM, Gemma2ForCausalLM, LlamaForCausalLM


def model_supported(model: AutoModelForCausalLM) -> Tuple[bool, str]:
    if isinstance(model, LlamaForCausalLM):
        return True, "Llama3"
    elif isinstance(model, Gemma2ForCausalLM):
        return True, "Gemma2"
    else:
        return False, None


def llama_post_process(text: str) -> str:
    text = text.split("<|end_header_id|>")[-1]
    text = text.replace("<|eot_id|>","")
    text = text.replace("<|begin_of_text|>", "")
    
    return text
