import guidance
import os

home_dir = os.path.expanduser("~")
with open(os.path.join(home_dir, "openai_api_key.txt")) as fin:
    line = fin.readline()
api_key = line.strip()

gpt = guidance.llms.OpenAI("gpt-4o-mini", api_key=api_key, chat_mode=True)

experts = guidance('''
{{#system~}}
You are a helpful and terse assistant.
{{~/system}}

{{#user~}}
I want a response to the following question:
{{query}}
Name 3 world-class experts (past or present) who would be great at answering this?
Don't answer the question yet.
{{~/user}}

{{#assistant~}}
{{gen 'expert_names' temperature=0 max_tokens=300}}
{{~/assistant}}

{{#user~}}
Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
{{~/user}}

{{#assistant~}}
{{gen 'answer' temperature=0 max_tokens=500}}
{{~/assistant}}
''', llm=gpt)

executed_program = experts(query='How can I be more productive?')
print(executed_program)
print(executed_program["expert_names"])
print(executed_program["answer"])


# Example output:

## executed_program["expert_names"]
# Sure! Here are three world-class experts who could provide valuable insights on productivity:                                                     

# 1. **David Allen** - Author of "Getting Things Done," a widely recognized productivity methodology.                                               
# 2. **Cal Newport** - Author of "Deep Work," focusing on the importance of focused work in achieving high productivity.                            
# 3. **Tim Ferriss** - Author of "The 4-Hour Workweek," known for his unconventional approaches to productivity and time management. 

## executed_program["answer"]
# To enhance your productivity, consider the following strategies:

# 1. **Implement a Systematic Approach**: Adopt a structured method like Getting Things Done (GTD). Break tasks into actionable steps, prioritize them, and regularly review your progress. This will help you manage your workload effectively and reduce overwhelm.                                  

# 2. **Cultivate Deep Work**: Dedicate uninterrupted blocks of time to focus on your most important tasks. Minimize distractions by creating a conducive work environment and setting clear boundaries. This will allow you to produce higher quality work in less time.                               

# 3. **Optimize Your Time and Energy**: Experiment with time management techniques, such as the Pomodoro Technique or time blocking, to find what works best for you. Additionally, assess your energy levels throughout the day and schedule demanding tasks during your peak performance times.      

# By integrating these principles, you can create a personalized productivity system that maximizes your efficiency and effectiveness. 