from abc import ABC, abstractmethod
import random 
import os

terminal_width = os.get_terminal_size().columns

class Runnable(ABC):

  @abstractmethod
  def invoke(input_data):
    pass

class DummyLLM(Runnable):

  def __init__(self):
    print('LLM created')

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
    
class DummyPromptTemplate(Runnable):

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def invoke(self, input_dict):
    return self.template.format(**input_dict)

  def format(self, input_dict):
    return self.template.format(**input_dict)

class DummyStrOutputParser(Runnable):

  def __init__(self):
    pass

  def invoke(self, input_data):
    return input_data['response']
  
class RunnableConnector(Runnable):

  def __init__(self, runnable_list):
    self.runnable_list = runnable_list

  def invoke(self, input_data):

    for runnable in self.runnable_list:
      input_data = runnable.invoke(input_data)

    return input_data

template = DummyPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = DummyLLM()

parser = DummyStrOutputParser()

chain = RunnableConnector([template, llm, parser])

result = chain.invoke({'length':'long', 'topic':'india'})

print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Simple Call to LLM")
print("=" * terminal_width)
print(template)
print(result)
print("=" * terminal_width)

template1 = DummyPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = DummyPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

llm = DummyLLM()

parser = DummyStrOutputParser()

chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])
final_chain = RunnableConnector([chain1, chain2])

result = final_chain.invoke({'topic':'cricket'})

print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Mimicing Chains")
print("=" * terminal_width)
print(result)
print("=" * terminal_width)