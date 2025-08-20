import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

terminal_width = os.get_terminal_size().columns

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='./assets\data/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)


print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "Using load()")
print("=" * terminal_width)
docs = loader.load()

for document in docs:
    print(document.metadata)
print("\n")
print("=" * terminal_width)

print("\n\n")
print("=" * terminal_width)
print((" " * ((terminal_width // 2) - 20)) + "lazy_load()")
print("=" * terminal_width)
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

print("=" * terminal_width)