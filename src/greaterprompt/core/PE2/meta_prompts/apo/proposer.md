{{#system~}}
You are a helpful assistant.
{{/system~}}

{{#user~}}
I'm trying to write a zero-shot classifier.

My current prompt is:
"{{prompt}}"

But it gets the following examples wrong:
{{failure_string}}

Based on these examples the problem with this prompt is that:
{{gradient}}

Based on the above information, I wrote an improved prompt. The total length of the prompt should be less than {{max_tokens}} words.
{{/user~}}

{{#assistant~}}
{{gen 'new_prompt' temperature=0.0}}
{{/assistant~}}