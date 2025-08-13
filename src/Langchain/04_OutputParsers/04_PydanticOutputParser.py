import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from getpass import getpass

from pydantic import BaseModel, Field

load_dotenv()

terminal_width = os.get_terminal_size().columns
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

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)


print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Use Case 1 Result")
print("=" * terminal_width)

prompt = template.invoke({'place': "Shilong"})
result = model.invoke(prompt)
parsed_result = parser.parse(result.content)

print("Prompt: ", prompt)
print("Result:", result.content)
print("Parsed Result:", parsed_result)
print("Parsed Result Type:", type(parsed_result))
print("=" * terminal_width)


print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Use Case 2 Result")
print("=" * terminal_width)

chain = template | model | parser

result = chain.invoke({'place': "Shilong"})

print("Prompt: ", prompt)
print("Result:", result)
print("Parsed Result Type:", type(result))
print("=" * terminal_width)