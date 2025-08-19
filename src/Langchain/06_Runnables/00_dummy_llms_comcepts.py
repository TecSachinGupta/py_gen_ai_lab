import random 
import os

terminal_width = os.get_terminal_size().columns

class DummyLLM:

    def __init__(self):
        print("Inside DummyLLM constructor")
        
    def invoke(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]

        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        return {'response': random.choice(response_list)}
    
class DummyPromptTemplate:

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def format(self, input_dict):
    return self.template.format(**input_dict)

template = DummyPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = DummyLLM()

prompt = template.format({'length':'short','topic':'india'})

result = llm.predict(prompt)

print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Simple Call to LLM")
print("=" * terminal_width)
print(prompt)
print(result)
print("=" * terminal_width)

class DummyLLMChain:

  def __init__(self, llm, prompt):
    self.llm = llm
    self.prompt = prompt

  def run(self, input_dict):

    final_prompt = self.prompt.format(input_dict)
    result = self.llm.predict(final_prompt)

    return result['response']

chain = DummyLLMChain(llm, template)

result = chain.run({'length':'short', 'topic': 'india'})

print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Mimicing Chain")
print("=" * terminal_width)
print(result)
print("=" * terminal_width)

