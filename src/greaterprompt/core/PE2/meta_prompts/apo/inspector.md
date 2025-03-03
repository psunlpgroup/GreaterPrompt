{{#system~}}
You are a helpful assistant.
{{/system~}}

{{#user~}}
I'm trying to write a zero-shot classifier prompt.

My current prompt is:
"{{prompt}}"

But this prompt gets the following examples wrong:
{{failure_string}}

Give {{n_reasons}} reasons why the prompt could have gotten these examples wrong. Do not include other text.
{{/user~}}

{{#assistant~}}
{{gen 'gradients' temperature=0.7}}
{{/assistant~}}