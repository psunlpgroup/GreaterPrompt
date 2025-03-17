{{#system~}}
You are a helpful assistant.
{{~/system}}
                                           
{{#user~}}
I gave a friend an instruction and {{n_demo}} inputs. The friend read the instruction and wrote an output for every one of the inputs.
Here are the input-output pairs:

{{demos}}

What was the instruction? It has to be less than {{max_tokens}} tokens.
{{~/user}}

{{#assistant~}}
The instruction was {{gen 'instruction' [[GENERATION_CONFIG]]}}
{{~/assistant}}