import os
import faiss
import re

from pathlib import Path
from typing import List, Optional
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# setting up environment and global variables
TERMINAL_WIDTH = os.get_terminal_size().columns

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HUGGINGFACEHUB_API_TOKEN:
    from getpass import getpass
    HUGGINGFACEHUB_API_TOKEN = getpass("Enter Hugging Face Token: ")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

os.environ["HF_HOME"] = "./.cache"
os.environ["HUGGINGFACE_HUB_CACHE "] = "./.cache"

class FileDownloader:
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
            print(f"Loaded {len(self.urls)} URLs from {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            self.urls = []
        except Exception as e:
            print(f"Error reading file: {e}")
            self.urls = []

    def download_files(self):
        """Download all PDF files from the URLs"""
        self.downloaded_files = []
        
        for i, url in enumerate(self.urls, 1):
            print(f"Downloading {i}/{len(self.urls)}: {url}")
            
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Generate temporary filename
                temp_filename = f"temp_pdf_{i}.pdf"
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
                    print(f"✓ Downloaded successfully")
                else:
                    temp_path.unlink()  # Delete invalid file
                    print(f"✗ Not a valid PDF file")
                    
            except Exception as e:
                print(f"✗ Failed to download: {e}")

    def smart_rename(self):
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
                
                print(f"Renamed: {file_info['original_name']} -> {new_path.name}")
                
            except Exception as e:
                print(f"Error renaming {file_info['temp_path']}: {e}")
                # Keep original temp name
                file_info['final_path'] = file_info['temp_path']
                file_info['final_name'] = file_info['temp_path'].name

    def execute(self, url_file_path = './urls.txt'):
        """Execute the complete download and rename process"""
        print("Starting PDF download and smart rename process...")
        print("=" * 50)
        
        # Step 1: Read URLs
        self.read_urls(url_file_path)
        if not self.urls:
            print("No URLs to process. Exiting.")
            return
            
        # Step 2: Download files
        print("\nDownloading files...")
        self.download_files()
        
        # Step 3: Smart rename
        print("\nRenaming files based on content...")
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
            print(f"Error extracting title: {e}")
            
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
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        
        successful = sum(1 for f in self.downloaded_files if f['success'])
        total = len(self.urls)
        
        print(f"Total URLs processed: {total}")
        print(f"Successful downloads: {successful}")
        print(f"Failed downloads: {total - successful}")
        
        if successful > 0:
            print(f"\nDownloaded files:")
            for file_info in self.downloaded_files:
                if file_info['success']:
                    final_name = file_info.get('final_name', file_info['original_name'])
                    print(f"  • {final_name}")

class FaissStore:
    def __init__(self, store_path: str = "./vector_store", embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.store_path = store_path
        self.embedding_model_name = embedding_model_name
        self.vector_store = None
        self.retriever = None
        
        # Initialize embeddings
        print(f"🔧 Initializing embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        print("===>> Embedding model initialized")

    def create_load_store(self, documents: Optional[List[Document]] = None) -> FAISS:
        """Create new store or load existing one. If both exist, merge documents with existing store."""
        if os.path.exists(self.store_path):
            print("📁 Found existing vector store, loading...")
            vector_store = self.load_store()
            
            if documents:
                print(f"➕ Adding {len(documents)} new documents to existing store...")
                self.add_documents(documents)
                self.save_store()
                print("✅ Documents merged and store updated")
            
            return vector_store
        elif documents:
            print("🆕 Creating new vector store...")
            vector_store = self.create_store(documents)
            self.save_store()
            return vector_store
        else:
            raise ValueError("No existing store found and no documents provided to create new store")

    def create_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents."""
        print(f"🗂️  Creating FAISS vector store from {len(documents)} documents...")
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            print(f"   ✅ Vector store created successfully")
            print(f"   📊 Index size: {self.vector_store.index.ntotal} vectors")
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return self.vector_store
            
        except Exception as e:
            print(f"   ❌ Error creating vector store: {e}")
            raise e

    def load_store(self):
        """Load FAISS vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(self.store_path, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            print(f"📁 Vector store loaded from: {self.store_path}")
            return self.vector_store
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return None

    def save_store(self):
        """Save FAISS vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.save_local(self.store_path)
        print(f"Vector store saved to: {self.store_path}")

    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        print(f"++++++ Adding {len(documents)} new documents to vector store...")
        try:
            self.vector_store.add_documents(documents)
            print(f"====>>> Documents added successfully")
            print(f"====>>> Updated index size: {self.vector_store.index.ntotal} vectors")
        except Exception as e:
            print(f" >><< Error adding documents: {e}")
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
            print(f"🗑️  Vector store deleted from: {self.store_path}")
            self.vector_store = None
            self.retriever = None
        else:
            print(f"❌ Store path does not exist: {self.store_path}")

    def get_store_info(self):
        """Get information about the current vector store."""
        if self.vector_store is None:
            print("❌ No vector store initialized")
            return None
        
        info = {
            "index_size": self.vector_store.index.ntotal,
            "embedding_model": self.embedding_model_name,
            "store_path": self.store_path,
            "dimension": self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else "Unknown"
        }
        
        print("📊 Vector Store Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        return info

class RAGSystem:

    def __init__(self):
        pass

    def load_documents(self):
        pass

    def split_docs(self):
        pass

    def create_llm(self):
        pass

    def create_chains(self):
        pass

class RAGApplication:
    def __init__(self):
        pass
