import random
import guidance

def deduplicate(nodes, prev_nodes=None):
    # deduplicate prompts (there is a slight chance that two prompts are the same...)
    unique_keys = set()
    deduplicated_list = []
    if prev_nodes: # from past time stamps
        for node in prev_nodes:
            unique_keys.add(node.prompt)
    for node in nodes:
        if node.prompt not in unique_keys:
            unique_keys.add(node.prompt)
            deduplicated_list.append(node)
    return deduplicated_list 

def clean_string(s):
    # issues with substrate llm api
    return s.replace("<|im_start|>", "").replace("<|im_end|>", "").strip('"').strip()

def pack_demo_string(batch):
    d = ""
    for i, row in batch.iterrows():
        if isinstance(row["label"], list):
            label = random.choice(row["label"])
        else:
            label = row["label"]
        d += """Input: {}\nOutput: {}\n\n""".format(
            row["input"], label)
    return d

def _load_prompt(datafile: str):
    with open(datafile, "r") as f:
        prompt = f.read()
    return prompt

def get_llm(llm_name):
    assert llm_name in ["openai_gpt35", "openai_gpt4", "openai_gpt4_turbo", "openai_gpt4o", "openai_gpt4o_mini"]
    
    if llm_name == "openai_gpt35":
        llm = guidance.llms.OpenAI("gpt-3.5-turbo", max_retries=100, max_calls_per_min=10)
    elif llm_name == "openai_gpt4": # we were using gpt-4-0613 for the experiments in the paper
        llm = guidance.llms.OpenAI("gpt-4", max_retries=100, max_calls_per_min=1)
    elif llm_name == "openai_gpt4_turbo": # we were using gpt-4-0125-preview for the experiments in the paper
        llm = guidance.llms.OpenAI("gpt-4-0125-preview", max_retries=100, max_calls_per_min=3, chat_mode=True)
    elif llm_name == "openai_gpt4o": 
        llm = guidance.llms.OpenAI("gpt-4o", max_retries=100, max_calls_per_min=3, chat_mode=True)    
    elif llm_name == "openai_gpt4o_mini": 
        llm = guidance.llms.OpenAI("gpt-4o-mini", max_retries=100, max_calls_per_min=10, chat_mode=True)    
    return llm
