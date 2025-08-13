import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

terminal_width = os.get_terminal_size().columns

llm = HuggingFacePipeline.from_model_id(
    model_id="google/gemma-3-1b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, date of birth and city of a fictional person. \n {addition_instruction}",
    input_variables=[],
    partial_variables={'addition_instruction': parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)
parsed_result = parser.parse(result.content)

print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Use Case 1 Result")
print("=" * terminal_width)
print("Prompt: ", prompt)
print("Result:", result.content)
print("Parsed Result:", parsed_result)
print("Parsed Result Type:", type(parsed_result))
print("=" * terminal_width)



print("\n\n")
print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Use Case 2 Result")
print("=" * terminal_width)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}. \n{addition_instruction} ',
    input_variables=['text'],
    partial_variables={'addition_instruction': parser.get_format_instructions()}
)

topic = 'black hole'

chain = template1 | model | template2 | model | parser

result = chain.invoke({'topic': topic})

print("Prompt: ", prompt)
print("Result:", result)
print("Parsed Result Type:", type(result))
print("=" * terminal_width)