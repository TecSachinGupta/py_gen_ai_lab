import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

load_dotenv()

# tool create

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))

multiply.name

multiply.description

multiply.args

# tool binding

llm = ChatOpenAI()

llm.invoke('hi')

llm_with_tools = llm.bind_tools([multiply])

llm_with_tools.invoke('Hi how are you')

query = HumanMessage('can you multiply 3 with 1000')

messages = [query]

messages

result = llm_with_tools.invoke(messages)

messages.append(result)

messages

tool_result = multiply.invoke(result.tool_calls[0])

tool_result

messages.append(tool_result)

messages

llm_with_tools.invoke(messages).content

# tool create
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

convert.args

get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})

# tool binding
llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]

messages

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

ai_message.tool_calls

import json

for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

messages

llm_with_tools.invoke(messages).content

from langchain.agents import initialize_agent, AgentType

# Step 5: Initialize the Agent ---
agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)

# --- Step 6: Run the Agent ---
user_query = "Hi how are you?"

response = agent_executor.invoke({"input": user_query})

