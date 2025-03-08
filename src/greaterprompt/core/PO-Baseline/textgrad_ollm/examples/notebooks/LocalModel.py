import sys
import os
from contextlib import contextmanager


parent_parent_folder = os.path.abspath(os.path.join(os.getcwd(), '../../..'))

# Construct the full path to the local 'textgrad' folder
local_textgrad_folder = os.path.join(parent_parent_folder, 'textgrad')

sys.path.insert(0, local_textgrad_folder)

print(sys.path)


from openai import OpenAI
from textgrad.engine.local_model_openai_api import ChatExternalClient
import textgrad as tg


# start a server with lm-studio and point it to the right address; here we use the default address.
#client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

engine = ChatExternalClient(client=None, model_string='meta-llama/Llama-2-7b-chat-hf')
tg.set_backward_engine(engine, override=True)

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 + 4(3)(2))) / 6
x = (7 ± √73) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

loss_system_prompt = tg.Variable("""You will evaluate a solution to a math question. 
Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.""",
                                 requires_grad=False,
                                 role_description="system prompt")

loss_fn = tg.TextLoss(loss_system_prompt)
optimizer = tg.TGD([solution])

loss = loss_fn(solution)
print("#### loss")
print(loss.value)

loss.backward()
optimizer.step()
print("#### solution final")
print(solution.value)