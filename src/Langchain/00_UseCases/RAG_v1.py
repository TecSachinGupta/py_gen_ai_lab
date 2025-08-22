import os
import faiss

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PDFMinerLoader, UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

TERMINAL_WIDTH = os.get_terminal_size().columns

# Step 1: Load all documents, here we have placed the files in a folder 
# and now we use directiry loader to load all documents

loader = DirectoryLoader(
    path='./assets/data/papers',
    glob='*v*.pdf',
    loader_cls=PDFMinerLoader,
    loader_kwargs={
        "mode": "page"
    }
)

docs = loader.load()

# Initialize the Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Perform the split
text_splits = text_splitter.split_documents(docs)

# em
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embedding_dim = len(embeddings.embed_query("Hello World, Welcome to Cricket Vector Store using FAISS."))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Index chunks
_ = vector_store.add_documents(documents=text_splits)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

