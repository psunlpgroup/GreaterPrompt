# PO-Baseline/textgrad_ollm Setup Instructions

- Clone the Repository
- Change directory to ```PO-Baseline/textgrad_ollm/```
- Install requirements ```pip install -r requirements.txt```
- Install latest transformers specific version: ```pip install transformers==4.38.0```. You may face some errors, but still you should see 4.38.0 version installed. If otherwise let me know.
- Alternatively with conda you can create an environment for cuda version 12.4 as follows: 
```
conda create -n textgrad_openllm python=3.9
conda activate textgrad_openllm
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install fschat==0.2.20
pip install transformers==4.38.0
```
(You may face errors, but it should still work. If not, for installing fschat and transformers, use ```--no-deps``` flag.)

- **IMPORTANT UPDATE**: For running gemma2-9b-it, you MUST install the following transformers version: ```pip install transformers==4.42.4``` 

- Get access from huggingface-hub
```
from huggingface_hub import login

# Define your Hugging Face token
hf_token = "YOUR_HUGGING_FACE_TOKEN"  # I can provide my personal token, if required

# Log in to Hugging Face
login(token=hf_token)
```
- Change directory to ```evaluation```
- Now run: ```CUDA_VISIBLE_DEVICES=0 python prompt_optimization.py --task=BBH_sports_understanding --run_validation --do_not_run_larger_model --evaluation_engine=meta-llama/Meta-Llama-3-8B-Instruct --test_engine=meta-llama/Meta-Llama-3-8B-Instruct``` as a test run. I expect some issues, but we can fix those. Let me know when anything breaks.
- Additional Note:  It is faster to run on a single large GPU than on multiple GPUs. You can enforce this with `CUDA_VISIBLE_DEVICES=0`. If you need to divide it between multiple GPUs, omit it.


Please let me know how it goes. If we can ensure one run smootly goes, we can run all tasks in a loop. 

