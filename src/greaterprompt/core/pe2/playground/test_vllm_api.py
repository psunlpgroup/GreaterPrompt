from openai import OpenAI

## In another console, run the following,
## python -u -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8001 --model mistralai/Mistral-7B-Instruct-v0.2 --gpu_memory_utilization 0.8

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

completion = client.completions.create(model="mistralai/Mistral-7B-Instruct-v0.2",
                                      prompt="San Francisco is a")
print("Completion result:", completion)

chat_response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)

## Expected Output
### Completion result: Completion(id='cmpl-adfdeec90cc8473eb6c29c10e2ada2cb', choices=[CompletionChoice(finish_reason='length', index=0, logprobs=None, text=' foodie paradise, with dozens of exciting, new restaurants opening each year.', stop_reason=None)], created=1722372949, model='mistralai/Mistral-7B-Instruct-v0.2', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=16, prompt_tokens=5, total_tokens=21))
### Chat response: ChatCompletion(id='chat-eb65f598e5d64444b312bcadb82e1460', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content=" Why don't scientists trust atoms?\n\nBecause they make up everything!", role='assistant', function_call=None, tool_calls=[]), stop_reason=None)], created=1722372949, model='mistralai/Mistral-7B-Instruct-v0.2', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=17, prompt_tokens=13, total_tokens=30))  