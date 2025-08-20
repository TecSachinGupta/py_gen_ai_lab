import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
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
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)


url = 'https://www.amazon.in/Apple-MacBook-13-inch-10-core-Unified/dp/B0DZDDQ429/ref=sr_1_3?crid=EP63MO2829V&dib=eyJ2IjoiMSJ9.qy57AGeIXgokG7cmnLzTx5bDDqD4y92FpGvIrJ2O-UJnutcqb6hvA3YuRlYc-BQZsrpDw_2J-XTcH3ztoIHntjRiW-JVI8QLGCPGirhI0MUQsB6XhmCRLhbGqW7Bqi9lZxz38ZP0gmrQLoXkPGlJNakTCGkYzerNWi8jQoFWEhqU0o_zZWz4hc4wF65qzasn4jFVSRfJf0K55jg3XeuFyH5fYX1QDt9qf7nLtx5736o.fdQeig3J7LlyyclZc6KwSjsiMXz9LA20xFsAq7nRwvY&dib_tag=se&keywords=mac&qid=1755668952&sprefix=mac%2Caps%2C248&sr=8-3&th=1'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))