# <div align="center"><h5>🤩 GreaterPrompt: A Python Toolkit for Prompt Optimization</h5><div>

<div align="center">
<a href="https://github.com/WenliangZhoushan/GreaterPrompt/blob/main/LICENSE" target="_blank"><img src=https://img.shields.io/badge/license-MIT-green></a>
<a href="https://pypi.org/project/greaterprompt/" target="_blank"><img src=https://img.shields.io/badge/Pypi-GreaterPrompt-orange></a>
<a href="https://arxiv.org/abs/2412.09722" target="_blank"><img src=https://img.shields.io/badge/arXiv-2412.09722-b31b1b.svg></a>
<a href="https://colab.research.google.com/drive/1yUPWSG6DuFFD0VIcbCTFdYpxrdT0-Z-f?usp=sharing" target="_blank"><img src=https://colab.research.google.com/assets/colab-badge.svg></a>
<a href="https://github.com/WenliangZhoushan/GreaterPrompt/pulls" target="_blank"><img src=https://img.shields.io/github/issues-pr/WenliangZhoushan/GreaterPrompt></a>
<a href="https://github.com/WenliangZhoushan/GreaterPrompt/issues" target="_blank"><img src=https://img.shields.io/github/issues/WenliangZhoushan/GreaterPrompt></a>
</div>

<h4 align="center">
<p>
<a href="#wrench-installation">Installation</a> |
<a href="#rocket-quick-start">Quick-Start</a> |
<a href="https://colab.research.google.com/drive/1yUPWSG6DuFFD0VIcbCTFdYpxrdT0-Z-f?usp=sharing" target="_blank">Colab-Examples</a> |
<a href="#book-input-format"> Input-Format</a> |
<a href="#robot-optimize-configs"> Optimize-Configs</a> |
<a href="#sparkles-features">Features</a> |
<a href="#art-greaterprompt-ui"> GreaterPrompt-UI</a>
</p>
</h4>

GreaterPrompt is a python toolkit for prompt optimization which only levarges small models to achieve a better performances by large models. Our toolkit includes 3 different optimizer and supports 2 models family now.

<p align="center">
<img src="./images/overview.png">
</p>

## :wrench: Installation

To get started with GreaterPormpt, you can simply install with pip:

```base
pip install greaterprompt
```

## :rocket: Quick Start

### Step 1: Build Input Dataloader

We support 2 methods to build dataloader, either load from a jsonl file or custom input

#### Method 1: load from a jsonl file

```python
from greaterprompt import GreaterDataloader

dataset1 = GreaterDataloader(data_path="./data/boolean_expressions.jsonl")
```

#### Method 2: custom input

```python
from greaterprompt import GreaterDataloader

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
```

### Step 2: Init the Model and Tokenizer (Only for GreaterPrompt)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda:0')
model.gradient_checkpointing_enable() #to save the cuda memory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
```

### Step 3: Set Optimize Configs

This step is totally Optional. For details about parameters, please refer to <a href="#robot-optimize-configs"> Optimize-Configs</a>

```python
from torch.nn import functional as F

# for GreaterOptimizer
greater_optimize_config = {
    "intersect_q": 5,
    "candidates_topk": 10,
    "loss_function": F.cross_entropy,
    "perplexity_loss": True,
    "perplexity_lambda": 0.2,
    "filter": True,
    "generate_config": {
        "do_sample": True,
        "temperature": 0.2,
        "max_new_tokens": 512,
    }
}

# for ApeOptimizer, ApoOptimizer, Pe2Optimizer
optimize_config = {
    "task_model": "openai_gpt35_turbo_instruct",
    "optim_model": "openai_gpt4_turbo",
}

# for TextGradOptimizer
textgrad_optimize_config = {
    "evaluation_engine": "meta-llama/Meta-Llama-3-8B-Instruct",
    "test_engine": "meta-llama/Meta-Llama-3-8B-Instruct",
    "device": "cuda:0"
}
```

### Step 4: Init the Optimizers you wanna use

```python
from greaterprompt import (
    ApeOptimizer, ApoOptimizer, GreaterOptimizer, GreaterDataloader, Pe2Optimizer, TextGradOptimizer
)

ape_optimizer = ApeOptimizer(
    optimize_config=optimize_config # Optional
)

apo_optimizer = ApoOptimizer(
    optimize_config=optimize_config # Optional
)

greater_optimizer = GreaterOptimizer(
    model=model,
    tokenizer=tokenizer,
    optimize_config=greater_optimize_config # Optional
)

pe2_optimizer = Pe2Optimizer(
    optimize_config=optimize_config # Optional
)

textgrad_optimizer = TextGradOptimizer(
    textgrad_optimize_config # Optional
)
```

### Step 5: Pass the Dataloader to Optimizer and Optimize

```python
ape_result = ape_optimizer.optimize(dataloader1, p_init="think step by step")
apo_result = apo_optimizer.optimize(dataloader2, p_init="think step by step")
greater_result = greater_optimizer.optimize(
    dataloader1,
    p_extractor="\nNext, only give the exact answer, no extract words or any punctuation:",
    rounds=80
)
pe2_result = pe2_optimizer.optimize(dataloader2, p_init="think step by step")
textgrad_result = textgrad_optimizer.optimize(dataloader1, p_init="think step by step")

# print results
print(f'ape_result: {ape_result}')
print(f'-' * 30)
print(f'apo_result: {apo_result}')
print(f'-' * 30)
print(f'greater_result: {greater_result}')
print(f'-' * 30)
print(f'pe2_result: {pe2_result}')
print(f'-' * 30)
print(f'textgrad_result: {textgrad_result}')

```

## :book: Input Format

If you wanna use jsonl file as the input, please make sure each line contains "question", "prompt" and "answer" three mandatory keys

```jsonl
{"id": "0", "question": "not ( True ) and ( True ) is", "prompt": "Use logical reasoning and think step by step.", "answer": "False"}
{"id": "1", "question": "True and not not ( not False ) is", "prompt": "Use logical reasoning and think step by step.", "answer": "True"}
{"id": "2", "question": "not True or False or ( False ) is", "prompt": "Use logical reasoning and think step by step.", "answer": "False"}
{"id": "3", "question": "False or not ( True ) and False is", "prompt": "Use logical reasoning and think step by step.", "answer": "False"}
{"id": "4", "question": "True or not False and True and False is", "prompt": "Use logical reasoning and think step by step.", "answer": "True"}
```

## :robot: Optimize Configs

<details>
<summary>ApeOptimizer, ApoOptimizer, Pe2Optimizer Parameters</summary>

* `task_model: str`, the LLM that performs the task with a prompt. You can use OpenAI instruct models like `openai_gpt35_turbo_instruct` or you can also use models supported in vLLM. The code currently supports `mistralai/Mistral-7B-Instruct-v0.2`, `mosaicml/mpt-7b-instruct`, `01-ai/Yi-6B`.
* `optim_model: str`, the LLM that performs prompt engineering. Currently the code supports `openai_gpt35`, `openai_gpt4`, `openai_gpt4_turbo`, `openai_gpt4o`, `openai_gpt4o_mini`

</details>

<details>
<summary>GreaterPrompt Parameters</summary>

* `intersect_q: int`, use how many question/prompt inpur pair to build a batch to get candidates.
* `candidates_topk: int`, sample how many candidates for each p_i.
* `loss_function: Callable[torch.Tensor, torch.Tensor] -> torch.Tensor`, the loss function used for the backward to get the gradients.
* `perplexity_loss: bool`, whether to enable the perplexity_loss.
* `perplexity_lambda: float`, if perplexity loss was enabled, its weight in the whole loss function.
* `filter: bool`, whether to filter the p* to make sure all prompts are human readable.
* `generate_config: dict`: configs used for transformer model's generation.

</details>

<details>
<summary>TextGradOptimizer Parameters</summary>

* `evaluation_engine: str`, currently we only support Llama3 family models.
* `test_engine: str`, currently we only support Llama3 family models.
* `device: str`, which device you wanna use to load the model.

</details>


## :sparkles: Features

For GreaterOptimizer, we support custom loss function in order to optimize different tasks, you could either use customed loss function or other pytorch loss functions!

```python
import torch
from torch.nn import functional as F

def dummy_loss_fn(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(y_pred, y)
    dummy_loss = 2 * loss + 1

    return dummy_loss

# use custom loss function
optimize_config = {
    "loss_function": dummy_loss_fn
}

# use other pytorch loss function
optimize_config = {
    "loss_function": F.nll_loss
}
```

### :art: GreaterPrompt UI

With GreaterPrompt-UI, you can easily and quickly configure and experience the supported optimize methods through our visual interface, it makes everything much more efficient!

Before using the UI interface, you have to clone the repo

```bash
git clone git@github.com:WenliangZhoushan/GreaterPrompt.git

cd Web
streamlit run Overview.py
```

## :bookmark: License

GreaterPrompt is licensed under the [<u>MIT License</u>](./LICENSE).

## :star2: Citation

Please kindly cite our work if helps your research:

```BibTex
TODO: paper not start writing yet.
```
