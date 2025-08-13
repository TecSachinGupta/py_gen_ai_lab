import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

topic = 'black hole'

terminal_width = os.get_terminal_size().columns

# Approach 1: Without StrOutputParser
prompt1 = template1.invoke({'topic': topic})
result1 = model.invoke(prompt1)
prompt2 = template2.invoke({'text':result1.content})
result2 = model.invoke(prompt2)
print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Approach 1 Result")
print("=" * terminal_width)
print(result2.content)
print("=" * terminal_width)
print("\n")

# Approach 2: With StrOutputParser
parser = StrOutputParser() 

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print("=" * terminal_width)
print(" "* ((terminal_width // 2) - 20) + "Approach 2 Result")
print("=" * terminal_width)
print(result)
print("=" * terminal_width)