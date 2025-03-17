# from openai import OpenAI
import openai
import os

home_dir = os.path.expanduser("~")
with open(os.path.join(home_dir, "openai_api_key.txt")) as fin:
    line = fin.readline()
api_key = line.strip()
print(api_key)

openai.api_key=api_key

output = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=False,
)
print(output)

## Example output:
# {
#   "id": "chatcmpl-xxx",
#   "object": "chat.completion",
#   "created": 1722380151,
#   "model": "gpt-4o-mini-2024-07-18",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "This is a test."
#       },
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 12,
#     "completion_tokens": 5,
#     "total_tokens": 17
#   },
#   "system_fingerprint": "xxx"
# }