import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)

from fastchat.model import get_conversation_template
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import re


# TODO adjust the parameter of path
PATH = '/scratch2/share/model_files/huggingface'


class ModelClass(object):
    def __init__(self, *, model_path, is_distributed=False, use_path=True, system_prompt="", device):
        self.is_distributed = is_distributed


        self.system_prompt = system_prompt

        if is_distributed:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.device = torch.device(f'cuda:{self.rank}')
            torch.cuda.set_device(self.device)
            if use_path:

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16,
                    trust_remote_code=True, device_map=device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16,
                    trust_remote_code=True, device_map=device)

            self.model_name_or_path = self.model.name_or_path

            self.model = DDP(self.model, device_ids=[self.rank])

        else:

            if use_path:

                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,
                                                              trust_remote_code=True, device_map=device,
                                                              cache_dir=PATH)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16,trust_remote_code=True, device_map=device)

            self.model_name_or_path = self.model.name_or_path

        # self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto",
        #                                                   cache_dir=PATH)
        #
        # self.model = self.model.bfloat16()
        self.model.eval()
        if 'llama' in model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=PATH, add_bos_token=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = PATH)
        #self.tokenizer.pad_token = "[PAD]"

        if 'Llama-2' in model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_id = -1

        elif 'Llama-3' in model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        elif 'gemma-2' in model_path:
            pass

        # forced model loading across all gpus
        # so that Llama2-70B running is possible

        print("model loaded successfully")
        self.template = get_conversation_template(model_path)

    def get_answer(self, prompt, system_prompt=None, output_only=True, length = 1000, temperature=1.0, top_p=0):
        if system_prompt:
            self.system_prompt = system_prompt
        if type(prompt) == type("string"):
            prompt = [prompt]
        length = 1024
        #print(len(prompt))

        prompts_list = []

        if 'Llama-2' in self.model_name_or_path:

            if self.system_prompt != "":
                self.system_prompt = f"<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n"
            for k in prompt:
                self.template.messages = []
                self.template.append_message(self.template.roles[0], f"{k}")
                self.template.append_message(self.template.roles[1], "")

                temp = self.template.get_prompt()

                pattern = """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"""



                temp = re.sub(pattern, self.system_prompt, temp)
                prompts_list.append(temp)
                #
                #
                # prompts_list.append(self.template.get_prompt())

            # Distribute data across all gpus
            if self.is_distributed:
                total_prompts = len(prompts_list)
                prompts_per_gpu = total_prompts // dist.get_world_size()
                start_idx = self.rank * prompts_per_gpu
                end_idx = start_idx + prompts_per_gpu if self.rank != dist.get_world_size() - 1 else total_prompts
                local_prompts = prompts_list[start_idx:end_idx]
                encoding = self.tokenizer.batch_encode_plus(local_prompts, padding=True, return_tensors="pt").to(
                    self.device)

            else:
                encoding = self.tokenizer.batch_encode_plus(prompts_list, padding=True, return_tensors="pt").to(self.model.device)
            #out = self.model.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens = length, temperature=temperature)

            # debugged
            if encoding.input_ids.shape[1] + 50 > length:
                length = encoding.input_ids.shape[1] + 50

            out = self.model.generate(encoding.input_ids, attention_mask=encoding.attention_mask, do_sample=False)

            #out = out[out.find(self.template.roles[1]) + len(self.template.roles[1])]
            outs = self.tokenizer.batch_decode(out.tolist(), skip_special_tokens=True)

            #return out[out.find(self.template.roles[1]) + len(self.template.roles[1]):]
            return [out[out.find(self.template.roles[1]) + len(self.template.roles[1]):] for out in outs]

        elif 'Llama-3' in self.model_name_or_path:
            if self.system_prompt != "":
                self.system_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{self.system_prompt}<|eot_id|>\n"
            for k in prompt:
                user_message = k
                temp = f"{self.system_prompt}<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                prompts_list.append(temp)

            if self.is_distributed:
                total_prompts = len(prompts_list)
                prompts_per_gpu = total_prompts // dist.get_world_size()
                start_idx = self.rank * prompts_per_gpu
                end_idx = start_idx + prompts_per_gpu if self.rank != dist.get_world_size() - 1 else total_prompts
                local_prompts = prompts_list[start_idx:end_idx]
                encoding = self.tokenizer.batch_encode_plus(local_prompts, padding=True, return_tensors="pt").to(
                    self.device)

            else:
                encoding = self.tokenizer.batch_encode_plus(prompts_list, padding=True, return_tensors="pt").to(
                    self.model.device)

            #encoding = self.tokenizer.batch_encode_plus(prompts_list, return_tensors="pt", padding=True).to(self.model.device)
            #out = self.model.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=length, temperature=temperature)
            out = self.model.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=length,
                                     do_sample=False)
            outs = self.tokenizer.batch_decode(out.tolist(), skip_special_tokens=True)

            #return [out[len(self.tokenizer.decode(enc)):] for out, enc in zip(outs, encoding['input_ids'])]
            return [out.split("assistant")[-1].strip() for out in outs]

        elif 'gemma-2' in self.model_name_or_path:
            if self.system_prompt != "":
                self.system_prompt = f"{self.system_prompt} "
            for k in prompt:
                #user_message = f"<|startoftext|>user\n{k}\n"
                user_message = f"{k}"
                temp = f"<start_of_turn>user\n{self.system_prompt}{user_message}<end_of_turn>\n<start_of_turn>model\n"
                prompts_list.append(temp)

            encoding = self.tokenizer.batch_encode_plus(prompts_list, padding=True, return_tensors="pt").to(self.model.device)

            out = self.model.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=length, do_sample=False)
            outs = self.tokenizer.batch_decode(out.tolist(), skip_special_tokens=True)
            return [out.split("model\n")[-1].strip() for out in outs]

        elif 'gemma' in self.model_name_or_path:
            for k in prompt:
                #prompt_t = f"""<start_of_turn>user\n{k}<end_of_turn>\n<start_of_turn>model"""
                prompt_t = f"{k}"
                prompts_list.append(prompt_t)

            if self.is_distributed:
                total_prompts = len(prompts_list)
                prompts_per_gpu = total_prompts // dist.get_world_size()
                start_idx = self.rank * prompts_per_gpu
                end_idx = start_idx + prompts_per_gpu if self.rank != dist.get_world_size() - 1 else total_prompts
                local_prompts = prompts_list[start_idx:end_idx]
                encoding = self.tokenizer.batch_encode_plus(local_prompts, padding=True, return_tensors="pt").to(
                    self.device)

            else:
                encoding = self.tokenizer.batch_encode_plus(prompts_list, padding=True, return_tensors="pt").to(
                    self.model.device)
            # encoding = self.tokenizer.batch_encode_plus(prompts_list, padding=True, return_tensors="pt").to(
            #     self.model.device)
            out = self.model.module.generate(encoding.input_ids, attention_mask=encoding.attention_mask, max_new_tokens=length,
                                      temperature=temperature)
            outs = self.tokenizer.batch_decode(out.tolist(), skip_special_tokens=True)

            return [out[len(self.tokenizer.decode(enc)):] for out, enc in zip(outs, encoding['input_ids'])]