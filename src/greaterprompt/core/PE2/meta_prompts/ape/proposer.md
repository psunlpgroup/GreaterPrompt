{{#system~}}
You are a helpful assistant.
{{~/system}}
                                           
{{#user~}}
Generate a variation of the following instruction while keeping the semantic meaning.

{{prompt}}

The new instruction has to be less than {{max_tokens}} words.
Reply with the new instruction. Do not include other text.
{{~/user}}

{{#assistant~}}
{{gen 'new_prompt' [[GENERATION_CONFIG]]}}
{{~/assistant}}