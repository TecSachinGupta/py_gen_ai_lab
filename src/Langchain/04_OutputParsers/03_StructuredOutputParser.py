import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from getpass import getpass

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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
    ResponseSchema(name='fact_4', description='Fact 4 about the topic'),
    ResponseSchema(name='fact_5', description='Fact 5 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 5 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt = template.invoke({'topic': "black hole"})
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



print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Use Case 2 Result")
print("=" * terminal_width)

chain = template | model | parser

result = chain.invoke({'topic': "black hole"})

print("Result:", result)
print("Parsed Result Type:", type(result))
print("=" * terminal_width)