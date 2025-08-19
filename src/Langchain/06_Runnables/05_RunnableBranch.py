import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch
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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))