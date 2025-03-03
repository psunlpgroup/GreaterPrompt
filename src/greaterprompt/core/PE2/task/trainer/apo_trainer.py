import random
import os
import guidance 

from .default_trainer import DefaultTrainer
from .utils import clean_string, _load_prompt
from .node import Node

class APOTrainer(DefaultTrainer):

    def prepare_meta_prompts(self):
        # initializer
        init_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "initializer.md"))
        generation_config = "n={} temperature={} max_tokens={}".format(self.args.init_n_prompts, self.args.init_temperature, self.args.prompt_max_tokens)
        prompt = init_prompt.replace("[[GENERATION_CONFIG]]", generation_config)
        self.initializer = guidance(prompt, llm=self.llm) # guidance program for initialization
        
        # inspector: ~ loss.backward()
        inspector_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "inspector.md"))
        self.inspector = guidance(inspector_prompt, llm=self.llm)

        # proposer: ~ optimizer.step()
        proposer_prompt = _load_prompt(os.path.join(self.args.meta_prompts_dir, "proposer.md"))
        self.proposer = guidance(proposer_prompt, llm=self.llm)
        
    def _update(self, node, batch, indices):
        failure_string = self._pack_batch(batch)

        guidance.llms.OpenAI.cache.clear()
        f = self.inspector(prompt=node.prompt,
                            failure_string=failure_string, n_reasons=5)
        gradient = clean_string(f["gradients"])

        f = self.proposer(prompt=node.prompt,
                          failure_string=failure_string, gradient=gradient, max_tokens=self.args.prompt_max_tokens)

        new_node = Node(
            timestamp=node.timestamp+1,
            id=node.id+"-{}".format(node.n_child),
            parent=node.id,
            prompt=clean_string(f["new_prompt"]),
            reasoning=gradient,
            optim_batch=indices,
        )
        node.n_child += 1

        return new_node