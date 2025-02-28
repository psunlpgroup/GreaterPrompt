from src.greaterprompt import GreaterOptimizer, GreaterDataloader

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example1, use jsonl file to build dataset
dataset1 = GreaterDataloader(data_path="./data/boolean_expressions.jsonl")

# init model and tokenzier
MODEL_PATH = "/scratch1/wmz5132/models/huggingface/gemma-2-9b-it"
DEVICE = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# optimizer config
optimize_config = {
    "intersect_q": 5,
    "candidates_topk": 10,
    "loss_function": F.cross_entropy,
    "perplexity_loss": True,
    "perplexity_lambda": 0.2,
    "filter": True,
    "generate_config": {
        "temperature": 0.2,
        "max_new_tokens": 1024,
    }
}

# init optimizer and optimize
optimizer = GreaterOptimizer(
    model=model,
    tokenizer=tokenizer,
    optimize_config=optimize_config
)
outputs = optimizer.optimize(
    inputs=dataset1, 
    # this extractor will be applied to all prompts inside the dataset
    p_extractor="\nNext, only give the exact answer, no extract words or any punctuation:",
    rounds=105
)

# print results
for question, p_stars in outputs.items():
    print(f'question: {question}')
    print(f'p_stars: {p_stars}')


# Example2, use custom inputs to build dataset
dataset2 = GreaterDataloader(custom_inputs=[
    {
        "question": "((-1 + 2 + 9 * 5) - (-2 + -4 + -4 * -7)) =", 
        "prompt": "Use logical reasoning and think step by step.", 
        "answer": "24"
    },
    {
        "question": "((-9 * -5 - 6 + -2) - (-8 - -6 * -3 * 1)) =",
        "prompt": "Use logical reasoning and think step by step.",
        "answer": "63"
     },
    {
        "question": "((3 * -3 * 6 + -5) - (-2 + -7 - 7 - -7)) =",
        "prompt": "Use logical reasoning and think step by step.",
        "answer": "-50"
    }
])
outputs = optimizer.optimize(
    inputs=dataset2, 
    # this extractor will be applied to all prompts inside the dataset
    p_extractor="\nNext, only give the exact answer, no extract words or any punctuation:",
    rounds=105
)

# print results
for question, p_stars in outputs.items():
    print(f'question: {question}')
    print(f'p_stars: {p_stars}')
