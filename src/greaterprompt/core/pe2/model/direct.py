import os
import tqdm
import retrying

import openai

from .model import AbstractModel
from .utils import _load_prompt, create_batches


FULL_PRMOPT = """
Instruction: {{instruction}}

Input: {{input}}
Output: 
"""

class DirectModel(AbstractModel):
    def load_prompt(self):
        self.full_prompt = FULL_PRMOPT
  
    @retrying.retry(
        stop_max_attempt_number=100, # Maximum number of attempts
        wait_exponential_multiplier=1000, wait_exponential_max=10000, 
        # Wait 2^x * 1000 milliseconds between each retry, up to 10 seconds, then 10 seconds afterwards
        retry_on_exception=lambda e: isinstance(e, Exception)
    )
    def _request_openai(self, queries):
        engine = "text-davinci-003" if self.args.task_model == "openai_td003" else "gpt-3.5-turbo-instruct"

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
        
        all_raw_output = [choice.text.strip() for choice in responses.choices]
        
        return all_raw_output
        
    
    def _run_batch(self, instruction, batch):
        all_inputs = batch["input"].tolist()
        all_queries = []
        
        prompt = self.full_prompt.replace("{{instruction}}", instruction)
        
        for i in range(len(all_inputs)):
            query = prompt.replace("{{input}}", all_inputs[i])
            all_queries.append(query)
                               
        # response = self._request_substrate(all_queries)
        # all_raw_output = [choice["text"] for choice in response["choices"]]
        all_raw_output = self._request_openai(all_queries)
        
        return all_raw_output
    
    def run(self, instruction, data):
        data = data.copy()
        batches = create_batches(data, batch_size=10)
        
        all_raw_output = []
        for batch in tqdm.tqdm(batches, desc="Batches"):
            batch_raw_output = self._run_batch(instruction, batch)
            all_raw_output += batch_raw_output

        data["raw_output"] = all_raw_output

        return data
    
    def run_till_enough_errors(self, instruction, data, task, target_n_errors):
        data = data.copy().sample(frac=1.0, random_state=42) # random shuffle
        batches = create_batches(data, batch_size=10)
        
        all_raw_output = []
        n_done = 0

        for batch in tqdm.tqdm(batches, desc="Batches"):
            batch_raw_output = self._run_batch(instruction, batch)
            
            n_done += len(batch)
            all_raw_output += batch_raw_output

            data_done = data.iloc[:n_done]
            data_done["raw_output"] = all_raw_output

            data_done, _ = task.evaluate(data_done)
            n_errors = (data_done['score'] == 0.0).sum()
            if n_errors >= target_n_errors:
                self.logger.info("Found {} model mistakes by running {} examples".format(n_errors, n_done))
                break

        return data_done
