import os

from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from getpass import getpass

load_dotenv()

TERMINAL_WIDTH = os.get_terminal_size().columns

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory='./chroma_langchain_db',
    collection_name='sample'
)

# add documents
docIds = vector_store.add_documents(docs)

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])

# search documents
selectedDoc = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

# search with similarity score
similarDocsQuery = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

# meta-data filtering
similarDocsFilter = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])# add documents
docIds = vector_store.add_documents(docs)

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "Inserted Docs Id")
print(docIds)
print("=" * TERMINAL_WIDTH)

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "All Docs")
print(allDocs)
print("=" * TERMINAL_WIDTH)

# search documents
selectedDoc = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2
)

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "Selectd Doc")
print(selectedDoc)
print("=" * TERMINAL_WIDTH)

# search with similarity score
similarDocsQuery = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "Select docs based on score")
print(similarDocsQuery)
print("=" * TERMINAL_WIDTH)

# meta-data filtering
similarDocsFilter = vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "Select docs based on metadata")
print(similarDocsFilter)
print("=" * TERMINAL_WIDTH)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='09a39dc6-3ba6-4ea7-927e-fdda591da5e4', document=updated_doc1)

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "All Docs after update")
print(allDocs)
print("=" * TERMINAL_WIDTH)

# delete document
vector_store.delete(ids=['09a39dc6-3ba6-4ea7-927e-fdda591da5e4'])

# view documents
allDocs = vector_store.get(include=['embeddings','documents', 'metadatas'])

print((" " * ((TERMINAL_WIDTH // 2) - 1)) + "All Docs aftern delete")
print(allDocs)
print("=" * TERMINAL_WIDTH)