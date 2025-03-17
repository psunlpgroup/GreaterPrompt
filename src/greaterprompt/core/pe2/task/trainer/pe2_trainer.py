import random
import os
import guidance 

from .default_trainer import DefaultTrainer
from .utils import clean_string, _load_prompt
from .node import Node

class PE2Trainer(DefaultTrainer):
    def prepare_meta_prompts(self):

        # initializer
        init_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "initializer.md"))
        generation_config = "n={} temperature={} max_tokens={}".format(self.args.init_n_prompts, self.args.init_temperature, self.args.prompt_max_tokens)
        prompt = init_prompt.replace("[[GENERATION_CONFIG]]", generation_config)
        self.initializer = guidance(prompt, llm=self.llm) # guidance program for initialization
        
        # proposer: ~ optimizer.step()
        proposer_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "proposer.md"))
        self.proposer = guidance(proposer_prompt, llm=self.llm)
        
        if self.args.optim_use_instruction:
            assert "{{instruction}}" in proposer_prompt
            self.instruction_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "instruction.md"))
        if self.args.optim_use_demonstrations:
            assert "{{demonstrations}}" in proposer_prompt
            self.demonstrations_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "demonstrations.md"))
        if self.args.optim_use_optim_tutorial:
            assert "{{optim_tutorial}}" in proposer_prompt
            self.optim_tutorial_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "optim_tutorial.md"))
        
    def _update(self, node, batch, optim_batch):
        examples = self._pack_batch(batch)

        guidance.llms.OpenAI.cache.clear()
        full_prompt = self.model.full_prompt.replace("{{input}}", "<input>").replace("{{instruction}}", node.prompt)
        # prepare input args
        proposer_args = {
            "prompt": node.prompt, "timestamp": node.timestamp, 
            "full_prompt": full_prompt, 
            "step_size": None,
            # placeholder with None so that guidance won't complain
            "batch_size": None, "failures": None,
            "history": None, "instruction": None, "demonstrations": None, "optim_tutorial": None,
            "max_tokens": self.args.prompt_max_tokens
        }
        
        proposer_args["batch_size"] = self.args.batch_size
        proposer_args["examples"] = examples
        if self.args.optim_use_step_size:
            proposer_args["step_size"] = self.args.step_size
        if self.args.optim_use_momentum:
            proposer_args["history"] = node.history
        if self.args.optim_use_instruction:
            proposer_args["instruction"] = self.instruction_prompt
        if self.args.optim_use_demonstrations:
            proposer_args["demonstrations"] = self.demonstrations_prompt
        if self.args.optim_use_optim_tutorial:
            proposer_args["optim_tutorial"] = self.optim_tutorial_prompt

        # guidance program
        f = self.proposer(**proposer_args)

        new_prompt = clean_string(f["new_prompt"])
        
        # post-process optimization history
        if self.args.optim_use_momentum:
            new_history = clean_string(f["new_history"])
            full_history = node.history + "\n" + new_history if node.timestamp > 0 else new_history
        else:
            full_history = None
            
        reasoning = clean_string(f["reasoning"]) + "[SEP]" + clean_string(f["reasoning2"]) if "reasoning2" in f else clean_string(f["reasoning"])
        new_node = Node(
            timestamp=node.timestamp+1,
            id=node.id+"-{}".format(node.n_child),
            parent=node.id,
            prompt=new_prompt,
            history=full_history,
            optim_batch=optim_batch,
            reasoning=reasoning
        )

        node.n_child += 1

        return new_node