from optimizer import GreaterOptimizer
from utils.dataloader import GreaterDataSet

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example1, use jsonl file to build dataset
dataset1 = GreaterDataSet(data_path="./data/boolean_expressions.jsonl")
dataset1.items = dataset1.items[:1]

# init model and tokenzier
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:6")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# optimizer config
optimize_config = {
    "intersection": False,
    "candidates_topk": 3,
    "generate_config": {
        "max_new_tokens": 1024
    }
}

# init optimizer and optimize
optimizer = GreaterOptimizer(
    model=model,
    tokenizer=tokenizer,
    optimize_config=optimize_config
)
p_stars = optimizer.optimize(
    inputs=dataset1, 
    # this extractor will be applied to all prompts inside the dataset
    extractor="\nNext, only give the exact answer, no extract words or any punctuation: ",
    rounds=35
)

# print results
for p_init, p_star in zip(dataset1, p_stars):
    print(f'p_init: {p_init["prompt"]}')
    print(f'p_stars: ')
    for i, p in enumerate(p_star):
        print(f'{i + 1}: {p}')


# Example2, use custom inputs to build dataset
dataset2 = GreaterDataSet(custom_inputs=[
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
p_stars = optimizer.optimize(
    inputs=dataset2, 
    # this extractor will be applied to all prompts inside the dataset
    extractor="\nNext, only give the exact answer, no extract words or any punctuation: ",
    rounds=35
)
for p_init, p_star in zip(dataset2, p_stars):
    print(f'p_init: {p_init["prompt"]}')
    print(f'p_stars: ')
    for i, p in enumerate(p_star):
        print(f'{i + 1}: {p}')
