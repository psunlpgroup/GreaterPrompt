## Prompt Engineering a Prompt Engineer

Code for paper "Prompt Engineering a Prompt Engineer" (https://arxiv.org/abs/2311.05661), to appear at ACL 2024 (Findings). In the paper,
* We investigate __what makes a good meta-prompt__ in LLM-powered automatic prompt engineering.
* We develop __PE2__, a strong automatic prompt engineer featuring __three meta-prompt components__.
* We show that PE2 can (1) make targeted and highly specific prompt edits; (2) induce multi-step plans for complex tasks; (3) reason and adapt in non-standard situations.

### Environment

```bash
conda create --name pe2 python=3.11
conda activate pe2
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

* We uploaded all preprocessed data files in `data`. 
* We included scripts to download and preprocess the data in `data/preprocess`.

### Run

Check out `example_scripts` to run experiments with PE2:
* Run `bbh_boolean_expressions.sh` for optimizing the prompt for the task of boolean expressions (from BIG-bench Hard), with `gpt-3.5-turbo-instruct` as the task model, and `gpt-4-turbo` as the optimization model.
* Run `bbh_boolean_expressions_vllm.sh` if you want to use a model supported in vLLM as the task model. We use `mistralai/Mistral-7B-Instruct-v0.2` here as an example.
* You can modify the `method` variable in the scripts to run baselines such as APE and APO.

<details>
<summary>Explanations on the arguments</summary>

* `--trainer`, select from "ape", "apo", "pe2". Each prompt optimization method uses slightly different workflows, and they are implemented in different "trainers".
* `--meta_prompts_dir`, directory pointing to the meta prompts that will be used.
* `--resume`, the code saves intermediate results to the output_dir. If this flag is used, you can rerun the same script to resume. This is helpful when the program crashes (e.g., due to hitting OpenAI rate limits)
* `--task` and `--subtask`, specify the task you would like to run prompt optimization
  * task = math, subtask = gsm8k, multiarith
  * task = ii, see subtask names in `data/instruction_induction`
  * task = cf, see subtask names in `data/counterfactual-evaluation`
  * task = bbh, see subtask names in `data/bbh`
* `--task_model`, the LLM that performs the task with a prompt.
  * You can use OpenAI instruct models like `openai_gpt35_turbo_instruct`.
  * You can also use models supported in vLLM. The code currently supports `mistralai/Mistral-7B-Instruct-v0.2`, `mosaicml/mpt-7b-instruct`, `01-ai/Yi-6B`.
  * You can add a new vLLM model by editing code in `model/zeroshotcot.py`.
* `--optim_model`, the LLM that performs prompt engineering.
  * Currently the code supports `openai_gpt35`, `openai_gpt4`, `openai_gpt4_turbo`, `openai_gpt4o`, `openai_gpt4o_mini`.
  * If OpenAI released a new model, you can add it by editing the `get_llm` function in `trainer/utils.py`. We integrated gpt4o and gpt4o-mini in this way and the code worked directly.
* `--init_method`, choose between "file" or "induction". If "file" is selected, use `--init_prompt_file` to specify the where the initiailzation prompt is. The code reads the file from the data directory.
</details>


### Contact Us
If you want to discuss about the paper or the code implementation, please reach out to Qinyuan (qinyuany@usc.edu)!

If you used our code in your study, or find our paper useful, please cite us using the following BibTeX:

<details>
<summary>BibTeX</summary>

```
@article{ye2023prompt,
  title={Prompt engineering a prompt engineer},
  author={Ye, Qinyuan and Axmed, Maxamed and Pryzant, Reid and Khani, Fereshte},
  journal={arXiv preprint arXiv:2311.05661},
  year={2023}
}
```
</details>
