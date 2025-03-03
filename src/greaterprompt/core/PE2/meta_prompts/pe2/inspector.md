{{#system~}}
You are a helpful assistant.
{{/system~}}

{{#user~}}
I am prompting large language models to do a task. Your job is to help me examine a prompt and a failure example, and provide feedback on how to improve the prompt.

Here is the prompt I am using.
{{prompt}}

The input of the example is:
{{input}}
                
The generated output by using the prompt is:
{{output}}

The golden label for this input is:
{{label}}

The golden label is absolutely correct. According to external evaluation, the generated output is not correct. This may be due to the prompt being not clear or precise.

Please examine the prompt and the example closely. Is the prompt describing the task reflected by the examples? How to improve the prompt so that the model will produce the correct output? Note that you should be open-minded and think about all possibilities when editing the prompt, since the examples may represent special and non-standard tasks (e.g., doing arithmetic operation with a different base).

Please provide detailed explanations and feedback on how to edit the prompt so it will output the golden label. After this, propose a better version of the prompt. 
{{/user~}}

{{#assistant~}}
{{gen 'feedback' temperature=0.0}}
{{/assistant~}}