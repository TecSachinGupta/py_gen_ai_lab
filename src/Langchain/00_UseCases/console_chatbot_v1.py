import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    task="text-generation",
    provider="auto", 
    temperature=0,
)

model = ChatHuggingFace(llm=llm)

chat_history = []
chat_history2 = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)

    print("AI: ",result.content)

    chat_history2.append({"prompt": user_input, "result": result.content})
