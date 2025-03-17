import os
import re
import tqdm
import retrying

import openai
from vllm import LLM, SamplingParams

from .model import AbstractModel
from .utils import create_batches, get_answer_cleansing_prompt

FULL_PRMOPT = """Q: {{input}}
A: {{instruction}}
"""

class ZeroshotCoTModel(AbstractModel):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        if any(model in self.args.task_model for model in ["Mistral-7B-Instruct-v0.2", "mpt-7b-instruct", "Yi-6B"]):
        # if self.args.task_model in ["mistralai/Mistral-7B-Instruct-v0.2", "mosaicml/mpt-7b-instruct", "01-ai/Yi-6B"]:
            logger.info("Setting up `{}` model in vLLM".format(self.args.task_model))
            self.llm = LLM(model=self.args.task_model)

    def load_prompt(self):
        self.full_prompt = FULL_PRMOPT
    
    # @retrying.retry(
    #     stop_max_attempt_number=100, # Maximum number of attempts
    #     wait_exponential_multiplier=1000, wait_exponential_max=10000, 
    #     # Wait 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
    #     retry_on_exception=lambda e: isinstance(e, Exception)
    # )
    def _request_instruct(self, queries):
        OPENAI_INSTRUCT_MODELS = {"openai_td003": "text-davinci-003", "openai_gpt35_turbo_instruct": "gpt-3.5-turbo-instruct"}
        engine = OPENAI_INSTRUCT_MODELS[self.args.task_model]

        responses = openai.Completion.create(
            engine=engine,
            prompt=queries,
            max_tokens=400,
            temperature=0.0,
            n=1,
            top_p=1,
            stream=False,
            stop=None
        )
        
        assert len(responses.choices) == len(queries)
        
        all_reasoning = [choice.text.strip() for choice in responses.choices]

        new_queries = []
        for query, choice in zip(queries, responses.choices):
            new_query = query + choice.text.strip() + " \n" + get_answer_cleansing_prompt(self.args)
            new_queries.append(new_query)

        responses = openai.Completion.create(
            engine=engine,
            prompt=new_queries,
            max_tokens=100 if self.args.subtask == "word_sorting" else 10, # word sorting answer is significantly longer
            temperature=0.0,
            n=1,
            top_p=1,
            stream=False,
            stop=None
        )

        assert len(responses.choices) == len(queries)

        all_raw_output = [choice.text.strip() for choice in responses.choices]

        return all_reasoning, all_raw_output

    @retrying.retry(
        stop_max_attempt_number=100, # Maximum number of attempts
        wait_exponential_multiplier=1000, wait_exponential_max=10000, 
        # Wait 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def _request_chat(self, queries):
        OPENAI_CHAT_MODELS = {"openai_gpt35_turbo": "gpt-3.5-turbo-1106", "openai_gpt4_turbo": "gpt-4-0125-preview"}
        engine = OPENAI_CHAT_MODELS[self.args.task_model]

        # client = OpenAI()
        all_reasoning, all_raw_output = [], []
        for query in queries:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                ],
                max_tokens=400,
                temperature=0.0,
                n=1,
                top_p=1,
                stream=False,
                stop=None
            )
            reasoning = response.choices[0].message.content
            all_reasoning.append(reasoning)

            # Second API call to do answer cleansing
            response = openai.ChatCompletion.create(
                model=engine,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": reasoning},
                    {"role": "user", "content": get_answer_cleansing_prompt(self.args)},
                ],
                max_tokens=10,
                temperature=0.0,
                n=1,
                top_p=1,
                stream=False,
                stop=None
            )
            final_answer = response.choices[0].message.content
            all_raw_output.append(final_answer)

        return all_reasoning, all_raw_output
    
    def _request_vllm(self, queries):
        all_reasoning, all_raw_output, new_queries = [], [], []

        # first call
        sampling_params = SamplingParams(temperature=0.0, 
                                         max_tokens=400, n=1,
                                         top_p=1, stop=None)
        responses = self.llm.generate(queries, sampling_params)

        for query, response in zip(queries, responses):
            reasoning = response.outputs[0].text
            all_reasoning.append(reasoning)
            new_query = query + reasoning + " \n" + get_answer_cleansing_prompt(self.args)
            new_queries.append(new_query)

        # second call
        responses = self.llm.generate(new_queries, sampling_params)
        max_tokens = 100 if self.args.subtask == "word_sorting" else 10
        sampling_params = SamplingParams(temperature=0.0, 
                                         max_tokens=max_tokens, n=1,
                                         top_p=1, stop=None)
        responses = self.llm.generate(new_queries, sampling_params)

        for response in responses:
            final_answer = response.outputs[0].text
            all_raw_output.append(final_answer)
        
        return all_reasoning, all_raw_output


    def _request(self, queries):
        if self.args.task_model in ["openai_td003", "openai_gpt35_turbo_instruct"]:
            return self._request_instruct(queries) # legacy openai completion endpoints
        elif self.args.task_model in ["openai_gpt35_turbo", "openai_gpt4_turbo"]:
            return self._request_chat(queries) # chat endpoints which may needs additional formating
        elif any(model in self.args.task_model for model in ["Mistral-7B-Instruct-v0.2", "mpt-7b-instruct", "Yi-6B"]):
            return self._request_vllm(queries)
        else:
            raise NotImplementedError

    def _run_batch(self, instruction, batch):
        all_inputs = batch["input"].tolist()
        all_queries = []
        
        prompt = self.full_prompt.replace("{{instruction}}", instruction)
        
        for i in range(len(all_inputs)):
            query = prompt.replace("{{input}}", all_inputs[i])
            all_queries.append(query)
                               
        # response = self._request_substrate(all_queries)
        # all_raw_output = [choice["text"] for choice in response["choices"]]
        all_reasoning, all_raw_output = self._request(all_queries)
        
        return all_reasoning, all_raw_output
    
    def run(self, instruction, data):

        # if self.args.task_model in ["mistralai/Mistral-7B-Instruct-v0.2", "mosaicml/mpt-7b-instruct", "01-ai/Yi-6B"]:
        #     openai.default_headers = {"x-foo": "true"}
        #     openai.api_key = "EMPTY"
        #     openai.api_base = "http://localhost:8001/v1"

        data = data.copy()
        batch_size = 10 if self.args.task_model in ["openai_td003", "openai_gpt35_turbo_instruct"] else len(data)
        batches = create_batches(data, batch_size=batch_size)
        
        all_raw_output = []
        all_reasoning = []
        for batch in tqdm.tqdm(batches, desc="Batches"):
            batch_reasoning, batch_raw_output = self._run_batch(instruction, batch)
            all_reasoning += batch_reasoning
            all_raw_output += batch_raw_output

        data["reasoning"] = all_reasoning
        data["raw_output"] = all_raw_output

        return data
    
    def run_till_enough_errors(self, instruction, data, task, target_n_errors):
        data = data.copy().sample(frac=1.0, random_state=42) # random shuffle
        batch_size = 10 if self.args.task_model in ["openai_td003", "openai_gpt35_turbo_instruct"] else 20
        batches = create_batches(data, batch_size=batch_size)
        
        all_raw_output = []
        all_reasoning = []
        n_done = 0

        for batch in tqdm.tqdm(batches, desc="Batches"):
            batch_reasoning, batch_raw_output = self._run_batch(instruction, batch)
            
            n_done += len(batch)
            all_reasoning += batch_reasoning
            all_raw_output += batch_raw_output

            data_done = data.head(n_done).copy()
            data_done["reasoning"] = all_reasoning
            data_done["raw_output"] = all_raw_output

            data_done, _ = task.evaluate(data_done)
            n_errors = (data_done['score'] == 0.0).sum()
            if n_errors >= target_n_errors:
                self.logger.info("Found {} model mistakes by running {} examples".format(n_errors, n_done))
                break

        return data_done
