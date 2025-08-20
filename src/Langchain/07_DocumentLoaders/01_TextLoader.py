import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

terminal_width = os.get_terminal_size().columns

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text-generation",
    provider="auto", 
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

textLoader = TextLoader("./assets/data/texts/cricket.txt", encoding="UTF-8")

docs = textLoader.load()

print("=" * terminal_width)

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

print("=" * terminal_width)

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

