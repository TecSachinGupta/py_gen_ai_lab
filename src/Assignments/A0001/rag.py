import os
import requests
import re
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque

# PDF Processing
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

# Evaluation imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from rouge_score import rouge_scorer

# Report generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv

load_dotenv()

# Setting up environment and global variables
TERMINAL_WIDTH = os.get_terminal_size().columns

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["HF_HOME"] = "./.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./.cache"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileDownloader:
    """Downloads PDF files from the specified ArXiv URLs"""
    
    def __init__(self, download_dir='./pdfs'):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.urls = []
        self.downloaded_files = []

    def read_urls(self, file_path = './urls.txt'):
        """Read URLs from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.urls = [line.strip() for line in file if line.strip() and not line.startswith('#')]
            logger.info(f"Loaded {len(self.urls)} URLs from {file_path}")
        except FileNotFoundError:
            logger.exception(f"File not found: {file_path}")
            self.urls = []
        except Exception as e:
            logger.exception(f"Error reading file: {e}")
            self.urls = []

    def download_files(self):
        """Download all PDF files from the URLs"""
        self.downloaded_files = []
        
        for i, url in enumerate(self.urls, 1):
            logger.info(f"Downloading {i}/{len(self.urls)}: {url}")
            
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Generate filename based on ArXiv ID
                arxiv_id = url.split('/')[-1].replace('.pdf', '')
                temp_filename = f"arxiv_{arxiv_id}.pdf"
                temp_path = self.download_dir / temp_filename
                
                # Download the file
                with open(temp_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                
                # Verify it's a PDF
                if self._is_valid_pdf(temp_path):
                    self.downloaded_files.append({
                        'url': url,
                        'temp_path': temp_path,
                        'original_name': temp_filename,
                        'success': True
                    })
                    logger.info(f"Downloaded successfully: {temp_filename}")
                else:
                    temp_path.unlink()  # Delete invalid file
                    logger.info(f"Not a valid PDF file")
                    
            except Exception as e:
                logger.info(f"Failed to download: {e}")

    def smart_rename(self):
        """Rename files based on extracted titles"""
        for file_info in self.downloaded_files:
            if not file_info['success']:
                continue
                
            try:
                # Extract title from PDF
                title = self._extract_title(file_info['temp_path'])
                
                if title:
                    # Clean title for filename
                    clean_title = self._clean_filename(title)
                    new_filename = f"{clean_title}.pdf"
                else:
                    # Fallback to URL-based name
                    new_filename = self._generate_filename_from_url(file_info['url'])
                
                # Handle duplicates
                new_path = self._get_unique_path(new_filename)
                
                # Rename file
                file_info['temp_path'].rename(new_path)
                file_info['final_path'] = new_path
                file_info['final_name'] = new_path.name
                
                logger.info(f"Renamed: {file_info['original_name']} -> {new_path.name}")
                
            except Exception as e:
                logger.exception(f"Error renaming {file_info['temp_path']}: {e}")
                # Keep original temp name
                file_info['final_path'] = file_info['temp_path']
                file_info['final_name'] = file_info['temp_path'].name

    def execute(self, url_file_path = './urls.txt'):
        """Execute the complete download and rename process"""
        logger.info("Starting PDF download and smart rename process...")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        # Step 1: Read URLs
        self.read_urls(url_file_path)
        if not self.urls:
            logger.info("No URLs to process. Exiting.")
            return
            
        # Step 2: Download files
        logger.info("Downloading files...")
        self.download_files()
        
        # Step 3: Smart rename
        logger.info("Renaming files based on content...")
        self.smart_rename()
        
        # Step 4: Summary
        self._print_summary()
        
    def _extract_title(self, pdf_path):
        """Extract title from PDF using pdfminer.six"""
        try:
            # First try to get title from PDF metadata
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                document = PDFDocument(parser)
                
                if document.info and len(document.info) > 0:
                    info = document.info[0]
                    if 'Title' in info:
                        title = info['Title']
                        if title and isinstance(title, bytes):
                            title = title.decode('utf-8', errors='ignore')
                        elif title:
                            title = str(title)
                        
                        if title and title.strip():
                            return title.strip()
            
            # If no metadata title, extract from first page content
            text = extract_text(pdf_path, maxpages=1)
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                # Look for title-like patterns (usually first few lines)
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    # Skip very short lines or lines that look like headers/footers
                    if len(line) > 10 and len(line) < 100:
                        # Avoid lines that are all caps (might be headers)
                        if not line.isupper():
                            return line
                
                # Fallback to first substantial line
                if lines:
                    return lines[0]
                    
        except Exception as e:
            logger.exception(f"Error extracting title: {e}")
            
        return None
        
    def _clean_filename(self, title):
        """Clean title to make it suitable for filename"""
        # Remove or replace invalid filename characters
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        title = re.sub(r'\s+', ' ', title)  # Multiple spaces to single space
        title = title.strip()
        
        # Truncate if too long
        if len(title) > 100:
            title = title[:100].rsplit(' ', 1)[0]  # Cut at word boundary
            
        return title if title else "untitled"
        
    def _generate_filename_from_url(self, url):
        """Generate filename from URL as fallback"""
        try:
            filename = os.path.basename(url.split('?')[0])  # Remove query params
            if filename.endswith('.pdf'):
                return filename
            return f"{filename}.pdf" if filename else "downloaded.pdf"
        except:
            return "downloaded.pdf"
            
    def _get_unique_path(self, filename):
        """Get unique file path, handling duplicates"""
        path = self.download_dir / filename
        counter = 1
        
        while path.exists():
            name_part, ext = os.path.splitext(filename)
            new_filename = f"{name_part}_{counter}{ext}"
            path = self.download_dir / new_filename
            counter += 1
            
        return path
        
    def _is_valid_pdf(self, file_path):
        """Check if file is a valid PDF"""
        try:
            with open(file_path, 'rb') as file:
                header = file.read(5)
                return header.startswith(b'%PDF-')
        except:
            return False
            
    def _print_summary(self):
        """Print download summary"""
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        successful = sum(1 for f in self.downloaded_files if f['success'])
        total = len(self.urls)
        
        logger.info(f"Total URLs processed: {total}")
        logger.info(f"Successful downloads: {successful}")
        logger.info(f"Failed downloads: {total - successful}")
        
        if successful > 0:
            logger.info(f"Downloaded files:")
            for file_info in self.downloaded_files:
                if file_info['success']:
                    final_name = file_info.get('final_name', file_info['original_name'])
                    logger.info(f"  • {final_name}")

class FaissStore:
    """Enhanced FAISS vector store with improved functionality"""
    
    def __init__(self, store_path: str = "./vector_store", embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.retriever = None
        
        # Initialize embeddings
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("Embedding model initialized")

    def create_load_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """Create new store or load existing one. If both exist, merge documents with existing store."""
        if os.path.exists(self.store_path):
            logger.info("Found existing vector store, loading...")
            vector_store = self.load_store()
            
            if documents:
                logger.info(f"Adding {len(documents)} new documents to existing store...")
                self.add_documents(documents)
                self.save_store()
                logger.info("Documents merged and store updated")
            
            return vector_store
        elif documents:
            logger.info("Creating new vector store...")
            vector_store = self.create_store(documents)
            self.save_store()
            return vector_store
        else:
            raise ValueError("No existing store found and no documents provided to create new store")

    def create_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        logger.info(f"Creating FAISS vector store from {len(documents)} documents...")
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info(f"Vector store created successfully")
            logger.info(f"Index size: {self.vector_store.index.ntotal} vectors")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return self.vector_store
            
        except Exception as e:
            logger.exception(f"Error creating vector store: {e}")
            raise e

    def load_store(self):
        """Load FAISS vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(
                self.store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            logger.info(f"Vector store loaded from: {self.store_path}")
            return self.vector_store
        except Exception as e:
            logger.exception(f"Error loading vector store: {e}")
            return None

    def save_store(self):
        """Save FAISS vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.save_local(self.store_path)
        logger.info(f"Vector store saved to: {self.store_path}")

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        logger.info(f"Adding {len(documents)} new documents to vector store...")
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Documents added successfully")
            logger.info(f"Updated index size: {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            logger.exception(f"Error adding documents: {e}")
            raise e

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Perform similarity search with scores."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """Get retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        return self.retriever

    def delete_store(self):
        """Delete the vector store from disk."""
        import shutil
        if os.path.exists(self.store_path):
            shutil.rmtree(self.store_path)
            logger.info(f"Vector store deleted from: {self.store_path}")
            self.vector_store = None
            self.retriever = None
        else:
            logger.info(f"Store path does not exist: {self.store_path}")

    def get_store_info(self):
        """Get information about the current vector store."""
        if self.vector_store is None:
            logger.info("No vector store initialized")
            return None
        
        info = {
            "index_size": self.vector_store.index.ntotal,
            "embedding_model": self.embedding_model_name,
            "store_path": self.store_path,
            "dimension": self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else "Unknown"
        }
        
        logger.info("Vector Store Information:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        return info

class ConversationMemory:
    """Manages conversation memory for the last 4 interactions"""
    
    def __init__(self, memory_size: int = 4):
        self.memory_size = memory_size
        self.conversations = deque(maxlen=memory_size)
        self.langchain_memory = ConversationBufferWindowMemory(
            k=memory_size,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def add_interaction(self, question: str, answer: str, source_documents: List = None):
        """Add a new interaction to memory"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'source_documents': source_documents or []
        }
        self.conversations.append(interaction)
        
        # Also add to LangChain memory
        self.langchain_memory.chat_memory.add_user_message(question)
        self.langchain_memory.chat_memory.add_ai_message(answer)
    
    def get_context_string(self) -> str:
        """Get formatted conversation history"""
        if not self.conversations:
            return ""
        
        context_parts = []
        for i, conv in enumerate(self.conversations):
            context_parts.append(f"Previous Q{i+1}: {conv['question']}")
            context_parts.append(f"Previous A{i+1}: {conv['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversations.clear()
        self.langchain_memory.clear()

class RAGSystem:
    """Main RAG System implementation"""

    def __init__(self, 
                 data_path: str = "./pdfs",
                 vector_store_path: str = "./vector_store",
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 llm_model: str = "microsoft/DialoGPT-medium",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.documents = []
        self.split_docs = []
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.conversational_chain = None
        self.memory = ConversationMemory()
        
        # Initialize FaissStore
        self.faiss_store = FaissStore(
            store_path=vector_store_path,
            embedding_model_name=embedding_model
        )
        
        logger.info("RAG System initialized")

    def load_documents(self) -> List[Document]:
        """Load documents from PDF files"""
        logger.info(f"Loading documents from: {self.data_path}")
        documents = []
        
        try:
            loader = DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyMuPDFLoader
            )
            self.documents = loader.load()
            logger.info(f"Loaded {len(self.documents)} PDF documents")
            
            # Add metadata for better tracking
            for i, doc in enumerate(self.documents):
                doc.metadata['doc_id'] = i
                doc.metadata['source_file'] = os.path.basename(doc.metadata.get('source', f'doc_{i}'))
                
            return self.documents
            
        except Exception as e:
            logger.exception(f"Error loading documents: {e}")
            raise e

    def split_documents(self) -> List[Document]:
        """Split documents into chunks"""
        if not self.documents:
            raise ValueError("No documents to split. Load documents first.")
        
        logger.info(f"Splitting documents into chunks...")
        logger.info(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[" ",
                            "\n## ",      # Section headers
                            "\n### ",     # Subsection headers  
                            "\n\n",       # Paragraph breaks
                            "\n",         # Line breaks
                            ". ",         # Sentence endings
                            " "           # Word boundaries
                ]
            )
            
            self.split_docs = text_splitter.split_documents(self.documents)
            logger.info(f"Created {len(self.split_docs)} chunks from {len(self.documents)} documents")
            
            return self.split_docs
            
        except Exception as e:
            logger.exception(f"Error splitting documents: {e}")
            return []

    def create_llm(self, **kwargs) -> Any:
        """Create LLM instance based on configuration."""
        logger.info(f"Creating LLM: {self.llm_model}")
        
        try:
            # Use a more conversational model suitable for QA
            endpoint = HuggingFacePipeline.from_model_id(
                model_id=self.llm_model,
                task='text-generation',
                pipeline_kwargs={
                    'max_new_tokens': kwargs.get("max_new_tokens", 200),
                    'temperature':kwargs.get("temperature", 0.7),
                    'top_p':kwargs.get("top_p", 0.9),
                    'repetition_penalty':kwargs.get("repetition_penalty", 1.1)
                }
            )
            
            self.llm = endpoint #ChatHuggingFace(llm=endpoint)
            logger.info("LLM created successfully")
            return self.llm  
        except Exception as e:
            logger.exception(f"Error creating LLM: {e}")
            # Fallback to a simpler model
            try:
                endpoint = HuggingFacePipeline.from_model_id(
                    model_id="google/flan-t5-base",
                    task='text2text-generation',
                    pipeline_kwargs={
                        'max_new_tokens': kwargs.get("max_new_tokens", 200),
                        'temperature':kwargs.get("temperature", 0.7),
                        'top_p':kwargs.get("top_p", 0.9),
                        'repetition_penalty':kwargs.get("repetition_penalty", 1.1)
                    }
                )
                self.llm = endpoint #ChatHuggingFace(llm=endpoint)
                logger.info("LLM created with fallback model")
                return self.llm
            except Exception as e2:
                logger.exception(f"Error creating fallback LLM: {e2}")
                raise e2

    def create_vector_store(self):
        """Create or load vector store"""
        if not self.split_docs:
            raise ValueError("No split documents available. Split documents first.")
        
        logger.info("Creating/loading vector store...")
        self.vector_store = self.faiss_store.create_load_store(self.split_docs)
        logger.info("Vector store ready")

    def create_chains(self):
        """Create different types of chains for RAG."""
        if self.llm is None:
            raise ValueError("LLM not initialized. Create LLM first.")
        
        if self.faiss_store.vector_store is None:
            raise ValueError("Vector store not initialized. Create vector store first.")
        
        logger.info("Creating chains...")

        # Custom prompt template for better responses
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        Provide a clear and informative answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Simple QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.faiss_store.get_retriever(k=3),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        # Conversational Chain with memory
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.faiss_store.get_retriever(k=3),
            memory=self.memory.langchain_memory,
            return_source_documents=True,
            verbose=False
        )
        
        logger.info("QA chains created successfully")

    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if use_memory and self.conversational_chain:
                result = self.conversational_chain.invoke({"question": question})
            elif self.qa_chain:
                result = self.qa_chain.invoke({"query": question})
            else:
                raise ValueError("No chains available. Create chains first.")
            
            # Extract answer and sources
            answer = result.get('answer', result.get('result', 'No answer generated'))
            source_documents = result.get('source_documents', [])
            
            # Add to memory
            self.memory.add_interaction(question, answer, source_documents)
            
            return {
                'question': question,
                'answer': answer,
                'source_documents': source_documents,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"Error querying RAG system: {e}")
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'source_documents': [],
                'timestamp': datetime.now().isoformat()
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system."""
        info = {
            "data_path": self.data_path,
            "documents_loaded": len(self.documents),
            "chunks_created": len(self.split_docs),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "vector_store_initialized": self.faiss_store.vector_store is not None,
            "qa_chain_initialized": self.qa_chain is not None,
            "conversational_chain_initialized": self.conversational_chain is not None
        }
        
        if self.faiss_store.vector_store:
            vector_info = self.faiss_store.get_store_info()
            if vector_info:
                info.update({f"vector_{k}": v for k, v in vector_info.items()})
        
        return info

class RAGEvaluator:
    """Evaluation framework for RAG system performance"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_relevance(self, question: str, retrieved_contexts: List[str]) -> float:
        """Evaluate context relevance using TF-IDF similarity."""
        if not retrieved_contexts:
            return 0.0
        
        try:
            # Combine question and contexts
            documents = [question] + retrieved_contexts
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarity between question and each context
            question_vector = tfidf_matrix[0:1]
            context_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(question_vector, context_vectors).flatten()
            return float(np.mean(similarities))
        except Exception as e:
            logger.warning(f"Error calculating relevance: {e}")
            return 0.5  # Default relevance score
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Evaluate answer faithfulness to context."""
        if not contexts or not answer:
            return 0.0
        
        try:
            context_text = " ".join(contexts)
            
            # Simple overlap-based faithfulness
            answer_words = set(answer.lower().split())
            context_words = set(context_text.lower().split())
            
            if not answer_words:
                return 0.0
            
            overlap = len(answer_words.intersection(context_words))
            faithfulness = overlap / len(answer_words)
            
            return min(faithfulness, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating faithfulness: {e}")
            return 0.0
    
    def evaluate_answer_quality(self, question: str, answer: str, ground_truth: str = None) -> Dict[str, float]:
        """Evaluate answer quality using multiple metrics."""
        metrics = {}
        
        try:
            # Readability score
            metrics['readability'] = min(textstat.flesch_reading_ease(answer) / 100.0, 1.0)
            
            # Length appropriateness (optimal around 50-100 words)
            word_count = len(answer.split())
            if word_count < 10:
                length_score = word_count / 10.0
            elif word_count <= 100:
                length_score = 1.0
            else:
                length_score = max(0.5, 100.0 / word_count)
            metrics['length_score'] = length_score
            
            # Completeness (does it seem to address the question)
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            completeness = len(question_words.intersection(answer_words)) / len(question_words) if question_words else 0.0
            metrics['completeness'] = completeness
            
            if ground_truth:
                # ROUGE scores
                rouge_scores = self.rouge_scorer.score(ground_truth, answer)
                metrics['rouge1_f'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2_f'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL_f'] = rouge_scores['rougeL'].fmeasure
                
        except Exception as e:
            logger.warning(f"Error calculating answer quality: {e}")
            # Provide default scores
            metrics.update({
                'readability': 0.5,
                'length_score': 0.5,
                'completeness': 0.5
            })
        
        return metrics
    
    def evaluate_contextual_awareness(self, current_question: str, current_answer: str, 
                                    conversation_history: List[Dict]) -> float:
        """Evaluate how well the bot uses conversation history."""
        if not conversation_history:
            return 0.0
        
        try:
            # Check if current answer references previous interactions
            prev_topics = set()
            for conv in conversation_history:
                prev_topics.update(conv['question'].lower().split())
                prev_topics.update(conv['answer'].lower().split())
            
            current_words = set(current_answer.lower().split())
            overlap_score = len(prev_topics.intersection(current_words)) / len(current_words) if current_words else 0.0
            
            return min(overlap_score * 2, 1.0)  # Amplify small overlaps
        except Exception as e:
            logger.warning(f"Error calculating contextual awareness: {e}")
            return 0.0
    
    def evaluate_rag_response(self, question: str, answer: str, contexts: List[str], 
                            conversation_history: List[Dict] = None, ground_truth: str = None) -> Dict[str, float]:
        """Comprehensive evaluation of RAG response."""
        evaluation = {
            'relevance': self.evaluate_relevance(question, contexts),
            'faithfulness': self.evaluate_faithfulness(answer, contexts),
            'contextual_awareness': self.evaluate_contextual_awareness(question, answer, conversation_history or [])
        }
        
        quality_metrics = self.evaluate_answer_quality(question, answer, ground_truth)
        evaluation.update(quality_metrics)
        
        # Overall score (weighted average)
        weights = {
            'relevance': 0.25,
            'faithfulness': 0.25,
            'contextual_awareness': 0.15,
            'readability': 0.15,
            'length_score': 0.1,
            'completeness': 0.1
        }
        
        overall_score = sum(evaluation[key] * weights[key] for key in weights if key in evaluation)
        evaluation['overall_score'] = overall_score
        
        return evaluation

class ReportGenerator:
    """Generate comprehensive PDF reports for RAG evaluation."""
    
    def __init__(self, output_path: str = "rag_evaluation_report.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles."""
        styles = {}
        
        # Title style
        styles['custom_title'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor='darkblue',
            alignment=1  # Center alignment
        )
        
        # Section header style
        styles['section_header'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor='darkblue'
        )
        
        # Subsection header style
        styles['subsection_header'] = ParagraphStyle(
            'SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor='darkred'
        )
        
        return styles
    
    def create_report(self, evaluation_results: Dict[str, Any], 
                     system_config: Dict[str, Any], 
                     test_questions: List[str]):
        """Create comprehensive PDF report."""
        doc = SimpleDocTemplate(self.output_path, pagesize=A4)
        story = []
        
        # Title page
        story.append(Paragraph("RAG System Evaluation Report", self.custom_styles['custom_title']))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['section_header']))
        summary_text = f"""
        This report presents the evaluation results of a Retrieval-Augmented Generation (RAG) system 
        designed to answer questions about recent advances in Natural Language Processing, specifically 
        focusing on transformer architectures, BERT, GPT-3, RoBERTa, and T5 models.
        <br/><br/>
        The system was tested with {len(test_questions)} predefined questions covering various aspects 
        of these models. The evaluation framework assessed multiple dimensions including relevance, 
        faithfulness, contextual awareness, and answer quality.
        <br/><br/>
        <b>Overall Performance Score: {evaluation_results.get('overall_performance', 0):.3f}/1.000</b>
        """
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # System Architecture
        story.append(Paragraph("System Architecture", self.custom_styles['section_header']))
        arch_text = f"""
        <b>Document Processing:</b> The system ingested 5 PDF documents from arXiv covering seminal 
        papers in NLP (Transformer, BERT, GPT-3, RoBERTa, T5).
        <br/><br/>
        <b>Text Chunking:</b> Documents were split into chunks of {system_config.get('chunk_size', 1000)} 
        tokens with {system_config.get('chunk_overlap', 200)} token overlap for better context preservation.
        <br/><br/>
        <b>Vector Database:</b> FAISS vector store with {system_config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2')} 
        embeddings for semantic similarity search.
        <br/><br/>
        <b>Language Model:</b> {system_config.get('llm_model', 'microsoft/DialoGPT-medium')} for response generation.
        <br/><br/>
        <b>Conversation Memory:</b> Maintains context for the last {system_config.get('memory_size', 4)} interactions.
        """
        story.append(Paragraph(arch_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Evaluation Methodology
        story.append(Paragraph("Evaluation Methodology", self.custom_styles['section_header']))
        method_text = """
        The evaluation framework assessed multiple dimensions:
        <br/><br/>
        <b>Relevance (25%):</b> TF-IDF cosine similarity between questions and retrieved contexts.
        <br/>
        <b>Faithfulness (25%):</b> Lexical overlap between generated answers and source contexts.
        <br/>
        <b>Contextual Awareness (15%):</b> Utilization of conversation history in responses.
        <br/>
        <b>Readability (15%):</b> Flesch reading ease score normalized to 0-1 range.
        <br/>
        <b>Length Appropriateness (10%):</b> Answer length optimization (target: 50-100 words).
        <br/>
        <b>Completeness (10%):</b> Coverage of question keywords in the answer.
        """
        story.append(Paragraph(method_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Results Summary
        story.append(Paragraph("Evaluation Results Summary", self.custom_styles['section_header']))
        
        if 'metric_averages' in evaluation_results:
            metrics = evaluation_results['metric_averages']
            results_data = [
                ['Metric', 'Average Score', 'Performance Level'],
                ['Relevance', f"{metrics.get('relevance', 0):.3f}", self._get_performance_level(metrics.get('relevance', 0))],
                ['Faithfulness', f"{metrics.get('faithfulness', 0):.3f}", self._get_performance_level(metrics.get('faithfulness', 0))],
                ['Contextual Awareness', f"{metrics.get('contextual_awareness', 0):.3f}", self._get_performance_level(metrics.get('contextual_awareness', 0))],
                ['Readability', f"{metrics.get('readability', 0):.3f}", self._get_performance_level(metrics.get('readability', 0))],
                ['Length Score', f"{metrics.get('length_score', 0):.3f}", self._get_performance_level(metrics.get('length_score', 0))],
                ['Completeness', f"{metrics.get('completeness', 0):.3f}", self._get_performance_level(metrics.get('completeness', 0))],
                ['Overall Score', f"{metrics.get('overall_score', 0):.3f}", self._get_performance_level(metrics.get('overall_score', 0))]
            ]
            
            table = Table(results_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), 'lightblue'),
                ('TEXTCOLOR', (0, 0), (-1, 0), 'black'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), 'lightgrey'),
                ('GRID', (0, 0), (-1, -1), 1, 'black'),
                ('FONTSIZE', (0, 1), (-1, -1), 10)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
        
        # Sample Question Analysis
        story.append(Paragraph("Sample Question Analysis", self.custom_styles['section_header']))
        
        if 'detailed_results' in evaluation_results:
            # Show analysis for top 3 questions
            for i, result in enumerate(evaluation_results['detailed_results'][:3]):
                story.append(Paragraph(f"Question {i+1} Analysis", self.custom_styles['subsection_header']))
                
                question_text = f"""
                <b>Question:</b> {result.get('question', 'N/A')}
                <br/><br/>
                <b>Generated Answer:</b> {result.get('answer', 'N/A')[:300]}{"..." if len(result.get('answer', '')) > 300 else ""}
                <br/><br/>
                <b>Evaluation Scores:</b>
                <br/>• Relevance: {result.get('evaluation', {}).get('relevance', 0):.3f}
                <br/>• Faithfulness: {result.get('evaluation', {}).get('faithfulness', 0):.3f}
                <br/>• Overall: {result.get('evaluation', {}).get('overall_score', 0):.3f}
                <br/>• Retrieved Contexts: {len(result.get('contexts', []))} chunks
                """
                story.append(Paragraph(question_text, self.styles['Normal']))
                story.append(Spacer(1, 15))
        
        # Performance Analysis
        story.append(Paragraph("Performance Analysis", self.custom_styles['section_header']))
        
        if 'metric_averages' in evaluation_results:
            metrics = evaluation_results['metric_averages']
            strengths = []
            weaknesses = []
            
            for metric, score in metrics.items():
                if score >= 0.7:
                    strengths.append(f"{metric.replace('_', ' ').title()}: {score:.3f}")
                elif score < 0.5:
                    weaknesses.append(f"{metric.replace('_', ' ').title()}: {score:.3f}")
            
            if strengths:
                story.append(Paragraph("Strengths", self.custom_styles['subsection_header']))
                strength_text = "<br/>".join([f"• {s}" for s in strengths])
                story.append(Paragraph(strength_text, self.styles['Normal']))
                story.append(Spacer(1, 10))
            
            if weaknesses:
                story.append(Paragraph("Areas for Improvement", self.custom_styles['subsection_header']))
                weakness_text = "<br/>".join([f"• {w}" for w in weaknesses])
                story.append(Paragraph(weakness_text, self.styles['Normal']))
                story.append(Spacer(1, 10))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['section_header']))
        recommendations_text = """
        Based on the evaluation results, the following improvements are recommended:
        <br/><br/>
        <b>1. Context Retrieval Enhancement:</b> Experiment with different embedding models 
        (e.g., sentence-transformers/all-mpnet-base-v2, OpenAI embeddings) to improve relevance scores.
        <br/><br/>
        <b>2. Response Generation Improvement:</b> Fine-tune the language model on domain-specific 
        data or use larger models (e.g., Llama-2, GPT-3.5) for better answer quality.
        <br/><br/>
        <b>3. Chunking Strategy Optimization:</b> Implement semantic chunking or adjust chunk 
        size/overlap parameters based on document structure.
        <br/><br/>
        <b>4. Memory Enhancement:</b> Implement more sophisticated conversation memory mechanisms 
        with attention-based context selection.
        <br/><br/>
        <b>5. Evaluation Framework Extension:</b> Incorporate human evaluation and domain expert 
        assessment for more comprehensive quality measurement.
        """
        story.append(Paragraph(recommendations_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Technical Details
        story.append(Paragraph("Technical Implementation Details", self.custom_styles['section_header']))
        tech_text = f"""
        <b>Documents Processed:</b> {system_config.get('num_documents', 5)} PDF files
        <br/>
        <b>Total Text Chunks:</b> {system_config.get('total_chunks', 'N/A')}
        <br/>
        <b>Vector Dimensions:</b> {system_config.get('vector_dimension', 'N/A')}
        <br/>
        <b>Average Query Time:</b> {evaluation_results.get('avg_query_time', 'N/A')} seconds
        <br/>
        <b>Memory Usage:</b> {evaluation_results.get('memory_usage', 'N/A')} MB
        """
        story.append(Paragraph(tech_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Conclusion
        story.append(Paragraph("Conclusion", self.custom_styles['section_header']))
        conclusion_text = f"""
        The RAG system demonstrates {self._get_performance_level(evaluation_results.get('overall_performance', 0)).lower()} 
        performance with an overall score of {evaluation_results.get('overall_performance', 0):.3f}. 
        The system successfully integrates document retrieval, semantic search, and conversational AI 
        to provide informative responses about advanced NLP topics.
        <br/><br/>
        Key achievements include effective document processing, semantic similarity matching, 
        and maintenance of conversational context. The evaluation framework provides comprehensive 
        assessment across multiple quality dimensions, enabling targeted improvements.
        <br/><br/>
        Future work should focus on enhancing answer faithfulness and contextual awareness 
        while maintaining the system's strengths in relevance and readability.
        """
        story.append(Paragraph(conclusion_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Comprehensive evaluation report generated: {self.output_path}")
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level description based on score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Adequate"
        else:
            return "Needs Improvement"

class RAGApplication:
    """Main RAG Application class that orchestrates the entire system"""
    
    def __init__(self, **kwargs):
        self.config: Dict = {
            'data_path': './pdfs',
            'vector_store_path': './vector_store',
            'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
            'llm_model': 'meta-llama/Llama-3.2-1B',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'retrieval_k': 3,
            'memory_size': 4
        }

        self.config.update(kwargs)

        # Predefined test questions (Assignment requirement: 10 questions)
        self.test_questions = [
            "What is the key innovation introduced by the Transformer architecture?",
            "How does self-attention work in the Transformer model?",
            "What are the main differences between BERT and GPT models?",
            "How does BERT's bidirectional training differ from traditional language models?",
            "What makes GPT-3 capable of few-shot learning?",
            "How does RoBERTa improve upon BERT's training methodology?",
            "What is the Text-to-Text Transfer Transformer (T5) approach?",
            "How do positional encodings work in Transformers?",
            "What are the advantages of the multi-head attention mechanism?",
            "How does the scaling of parameters affect language model performance?"
        ]

        self.rag_system = None
        self.evaluator = RAGEvaluator()
        self.test_results = []
        self.report_generator = ReportGenerator()

    def setup_system(self):
        """Setup the complete RAG system"""
        logger.info("Setting up RAG system...")
        
        # Step 1: Download PDFs if needed
        if not os.path.exists(self.config['data_path']) or len(os.listdir(self.config['data_path'])) == 0:
            logger.info("Downloading required PDF documents...")
            downloader = FileDownloader(self.config['data_path'])
            downloader.execute()
        
        # Step 2: Initialize RAG System
        self.rag_system = RAGSystem(
            data_path=self.config['data_path'],
            vector_store_path=self.config['vector_store_path'],
            embedding_model=self.config['embedding_model'],
            llm_model=self.config['llm_model'],
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        
        # Step 3: Load and process documents
        logger.info("Loading and processing documents...")
        self.rag_system.load_documents()
        self.rag_system.split_documents()
        
        # Step 4: Create vector store
        logger.info("Creating vector store...")
        self.rag_system.create_vector_store()
        
        # Step 5: Initialize LLM
        logger.info("Initializing language model...")
        self.rag_system.create_llm()
        
        # Step 6: Create QA chains
        logger.info("Creating QA chains...")
        self.rag_system.create_chains()
        
        logger.info("RAG system setup complete!")

    def run_evaluation(self):
        """Run comprehensive evaluation with the test questions"""
        if not self.rag_system:
            raise ValueError("RAG system not initialized. Run setup_system() first.")
        
        logger.info(f"Starting evaluation with {len(self.test_questions)} test questions...")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        self.test_results = []
        all_metrics = []
        start_time = datetime.now()
        
        for i, question in enumerate(self.test_questions, 1):
            logger.info(f"Question {i}/{len(self.test_questions)}: {question}...")
            
            # Query the RAG system
            query_start = datetime.now()
            response = self.rag_system.query(question, use_memory=True)
            query_time = (datetime.now() - query_start).total_seconds()
            
            # Extract context texts
            contexts = []
            if response['source_documents']:
                contexts = [doc.page_content for doc in response['source_documents']]
            
            # Get conversation history
            conversation_history = list(self.rag_system.memory.conversations)
            
            # Evaluate response
            evaluation = self.evaluator.evaluate_rag_response(
                question=question,
                answer=response['answer'],
                contexts=contexts,
                conversation_history=conversation_history[:-1]  # Exclude current interaction
            )
            
            # Store results
            result = {
                'question_id': i,
                'question': question,
                'answer': response['answer'],
                'contexts': contexts,
                'source_documents': response['source_documents'],
                'evaluation': evaluation,
                'query_time': query_time,
                'timestamp': response['timestamp']
            }
            
            self.test_results.append(result)
            all_metrics.append(evaluation)
            
            # Print progress
            logger.info(f"   Answer: {response['answer']}...")
            logger.info(f"   Overall Score: {evaluation.get('overall_score', 0):.3f}")
            logger.info(f"   Query Time: {query_time:.2f}s")
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate aggregate metrics
        metric_keys = ['relevance', 'faithfulness', 'contextual_awareness', 'readability', 
                      'length_score', 'completeness', 'overall_score']
        metric_averages = {}
        
        for key in metric_keys:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                metric_averages[key] = np.mean(values)
                metric_averages[f"{key}_std"] = np.std(values)
        
        # Store evaluation summary
        self.evaluation_summary = {
            'total_questions': len(self.test_questions),
            'total_time': total_time,
            'avg_query_time': total_time / len(self.test_questions),
            'metric_averages': metric_averages,
            'overall_performance': metric_averages.get('overall_score', 0),
            'detailed_results': self.test_results
        }
        
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info(f"Total Questions: {len(self.test_questions)}")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Average Query Time: {total_time/len(self.test_questions):.2f} seconds")
        logger.info(f"Overall Performance: {metric_averages.get('overall_score', 0):.3f}/1.000")
        
        logger.info("Metric Summary:")
        for metric, value in metric_averages.items():
            if not metric.endswith('_std'):
                std_val = metric_averages.get(f"{metric}_std", 0)
                logger.info(f"  {metric.replace('_', ' ').title()}: {value:.3f} (±{std_val:.3f})")

    def generate_report(self, output_path: str = "rag_evaluation_report.pdf"):
        """Generate comprehensive evaluation report"""
        if not hasattr(self, 'evaluation_summary'):
            raise ValueError("No evaluation results available. Run run_evaluation() first.")
        
        logger.info(f"Generating comprehensive evaluation report...")
        
        # Prepare system configuration for report
        system_config = self.config.copy()
        system_info = self.rag_system.get_system_info()
        system_config.update({
            'num_documents': system_info.get('documents_loaded', 0),
            'total_chunks': system_info.get('chunks_created', 0),
            'vector_dimension': system_info.get('vector_dimension', 'N/A')
        })
        system_config.update({
            'avg_query_time': self.evaluation_summary.get('avg_query_time', 0),
            'memory_usage': 'N/A'  # Could be implemented with psutil
        })
        
        # Generate report
        report_gen = ReportGenerator(output_path)
        report_gen.create_report(
            evaluation_results=self.evaluation_summary,
            system_config=system_config,
            test_questions=self.test_questions
        )
        
        logger.info(f"Report generated successfully: {output_path}")

    def interactive_mode(self):
        """Run interactive question-answering mode"""
        if not self.rag_system:
            raise ValueError("RAG system not initialized. Run setup_system() first.")
        
        logger.info("Starting interactive mode. Type 'quit' to exit.")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                # Query system
                response = self.rag_system.query(question, use_memory=True)
                
                # Display response
                logger.info(f"Answer: {response['answer']}")
                
                if response['source_documents']:
                    logger.info(f"Sources ({len(response['source_documents'])}):")
                    for i, doc in enumerate(response['source_documents'][:3], 1):
                        source = doc.metadata.get('source_file', f'Document {i}')
                        logger.info(f"  {i}. {source}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(f"Error in interactive mode: {e}")
        
        logger.info("Interactive mode ended.")

    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        logger.info("Starting complete RAG evaluation pipeline...")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        
        try:
            # Step 1: Setup system
            self.setup_system()
            
            # Step 2: Run evaluation
            self.run_evaluation()
            
            # Step 3: Generate report
            self.generate_report()
            
            # Step 4: Print summary
            self.print_final_summary()
            
            logger.info("=" * (TERMINAL_WIDTH - 50))
            logger.info("EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * (TERMINAL_WIDTH - 50))
            
            return self.evaluation_summary
            
        except Exception as e:
            logger.exception(f"Error in evaluation pipeline: {e}")
            raise

    def print_final_summary(self):
        """Print final evaluation summary"""
        if not hasattr(self, 'evaluation_summary'):
            return
        
        summary = self.evaluation_summary
        
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info("FINAL EVALUATION SUMMARY")
        logger.info("=" * (TERMINAL_WIDTH - 50))
        logger.info(f"Overall Performance Score: {summary['overall_performance']:.3f}/1.000")
        logger.info(f"Total Questions Processed: {summary['total_questions']}")
        logger.info(f"Average Response Time: {summary['avg_query_time']:.2f} seconds")
        
        logger.info("Top Performing Areas:")
        metrics = summary['metric_averages']
        sorted_metrics = sorted([(k, v) for k, v in metrics.items() if not k.endswith('_std')], 
                               key=lambda x: x[1], reverse=True)[:3]
        for metric, score in sorted_metrics:
            logger.info(f"  • {metric.replace('_', ' ').title()}: {score:.3f}")
        
        logger.info("Report Generated: rag_evaluation_report.pdf")
        logger.info("System Ready for Interactive Use!")


def main():
    """Main function to run the RAG assignment"""
    logger.info("Starting RAG Assignment Implementation")
    logger.info("=" * (TERMINAL_WIDTH - 50))
    try:
        # Initialize application
        app = RAGApplication(
            chunk_size=900,
            chunk_overlap=120,
            retrieval_k=3,
            memory_size=4,
            llm_model="microsoft/Phi-3-mini-4k-instruct" # google/gemma-3-270m-it 
        )
    except Exception as e:
        logger.exception(f"Application Initialization failed: {e}")
        raise e
    
    try:
        # Run complete evaluation
        results = app.run_complete_evaluation()
        
        # Optional: Start interactive mode
        user_input = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
        if user_input == 'y':
            app.interactive_mode()
        
        return results
        
    except Exception as e:
        logger.exception(f"Application failed: {e}")
        raise e

if __name__ == "__main__":
    results = main()