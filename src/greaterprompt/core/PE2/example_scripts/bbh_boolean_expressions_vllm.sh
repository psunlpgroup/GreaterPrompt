cd ..

subtask="boolean_expressions"
method="pe2" # select from "ape", "apo", "pe2"

python cli.py \
 --do_train --do_validate --do_test \
 --trainer ${method} --backtrack --resume \
 --meta_prompts_dir meta_prompts/${method} \
 --model zeroshotcot \
 --task bbh --subtask ${subtask} \
 --data_dir data/bbh/${subtask} \
 --output_dir output/bbh/${subtask}/${method}_mistral-7b-instruct/ \
 --task_model mistralai/Mistral-7B-Instruct-v0.2 \
 --optim_model openai_gpt4_turbo \
 --n_beam 4 \
 --n_expand 4 \
 --train_steps 4 \
 --init_temperature 0.7 \
 --init_method file \
 --init_prompt_file prompt.md;
