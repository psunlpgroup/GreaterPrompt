from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
# Test single-GPU inference
llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

## Expected Output:
### Prompt: 'Hello, my name is', Generated text: ' J.C. and I am a student at the University of California, Berkeley'                                                                           
### Prompt: 'The president of the United States is', Generated text: " not a racist. He is a racist.\nHe's a racist because he"                                                                 
### Prompt: 'The capital of France is', Generated text: ' the capital of the French Republic.\n\nThe capital of France is the capital'                                                          
### Prompt: 'The future of AI is', Generated text: ' in the hands of the people.\n\nThe future of AI is in the'  