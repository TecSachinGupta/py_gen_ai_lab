from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'SequentialCustomMetaData'

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatOpenAI(model='gpt-5-mini', temperature=0.7)
model2 = ChatOpenAI(model='gpt-4.1-mini', temperature=0.3)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

meta_configs = {
    "run_name": "sequential chain",
    "tags": ['llm app', 'report generation', 'summarization'],
    "metadata": {"model1": "gpt-5-mini", "model1_temprature":0.7, "model2": "gpt-4.1-mini", "model2_temprature":0.3, "parser":"String Output Parser"}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=meta_configs)

print(result)
