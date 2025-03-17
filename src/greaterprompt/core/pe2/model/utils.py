import guidance
import openai

def get_llm(llm_name):
    assert llm_name in ["openai_gpt35", "openai_gpt4", "openai_td003", "openai_gpt35_turbo_instruct"]
    
    if llm_name == "openai_gpt35":
        llm = guidance.llms.OpenAI("gpt-3.5-turbo", max_retries=100, max_calls_per_min=10)
    if llm_name == "openai_gpt35_turbo_instruct":
        llm = guidance.llms.OpenAI("gpt-3.5-turbo-instruct", max_retries=100, max_calls_per_min=10)
    elif llm_name == "openai_td003":
        llm = guidance.llms.OpenAI("text-davinci-003", max_retries=100, max_calls_per_min=10)
    elif llm_name == "openai_gpt4":
        llm = guidance.llms.OpenAI("gpt-4", max_retries=100, max_calls_per_min=5)

    return llm

def create_batches(df, batch_size):
    # Calculate the number of batches
    num_batches = len(df) // batch_size

    # Create batches
    batches = [df.iloc[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

    # If there are remaining rows not included in full batches, you can add them to the last batch
    remaining_rows = len(df) % batch_size
    if remaining_rows > 0:
        batches.append(df.iloc[-remaining_rows:])
        
    return batches

def _load_prompt(datafile: str):
    with open(datafile, "r") as f:
        prompt = f.read()
    return prompt

def clean_string(s):
    # issues with substrate llm api
    return s.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

def last_boxed_only_string(string):
    # from https://github.com/hendrycks/math/blob/main/modeling/dataset/util.py
    # for MATH dataset
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def get_answer_cleansing_prompt(args):
    if args.task == "math":
        prompt = "Therefore, the answer (arabic numerals) is"
    elif args.task == "bbh":
        if args.subtask == "formal_fallacies":
            prompt = "Therefore, the answer (valid or invalid) is"
        elif args.subtask in ["causal_judgement", "sports_understanding", "web_of_lies", "navigate"]:
            prompt = "Therefore, the answer (yes or no) is"
        else:
            prompt = "Therefore, the answer is"
    elif args.task == "custom":
        prompt = "Therefore, the answer is"
    else:
        raise NotImplementedError
    return prompt