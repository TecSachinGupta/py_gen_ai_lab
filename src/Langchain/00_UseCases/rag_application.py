import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# LangChain Core Components
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain.schema import Document

# HuggingFace Integration
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
import torch
from sentence_transformers import SentenceTransformer, util

# Evaluation Metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat
import nltk

# PDF and Report Generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class LangChainPDFIngestion:
    """
    PDF ingestion pipeline using LangChain document loaders.
    Handles the 5 specified arXiv papers with advanced processing.
    """
    
    def __init__(self):
        self.pdf_urls = [
            "https://arxiv.org/pdf/1706.03762.pdf",  # Attention Is All You Need
            "https://arxiv.org/pdf/1810.04805.pdf",  # BERT
            "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3
            "https://arxiv.org/pdf/1907.11692.pdf",  # RoBERTa
            "https://arxiv.org/pdf/1910.10683.pdf"   # T5
        ]
        self.pdf_names = [
            "attention_is_all_you_need.pdf",
            "bert.pdf",
            "gpt3.pdf", 
            "roberta.pdf",
            "t5.pdf"
        ]
        self.paper_titles = [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "Language Models are Few-Shot Learners (GPT-3)",
            "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "Exploring the Limits of Transfer Learning with T5"
        ]
        self.documents = []
        
    def download_pdfs(self) -> List[str]:
        """Download PDFs if not already present."""
        downloaded_paths = []
        
        for url, filename in zip(self.pdf_urls, self.pdf_names):
            if not os.path.exists(filename):
                print(f"📥 Downloading {filename}...")
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"✅ Successfully downloaded {filename}")
                except Exception as e:
                    print(f"❌ Error downloading {filename}: {e}")
                    continue
            else:
                print(f"📄 Using existing {filename}")
            
            downloaded_paths.append(filename)
        
        return downloaded_paths
    
    def load_and_process_documents(self) -> List[Document]:
        """Load PDFs using LangChain PyPDFLoader and process them."""
        pdf_paths = self.download_pdfs()
        all_documents = []
        
        for idx, pdf_path in enumerate(pdf_paths):
            print(f"\n📚 Processing document {idx + 1}/{len(pdf_paths)}: {pdf_path}")
            
            try:
                # Use LangChain PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Add metadata to documents
                for doc_idx, doc in enumerate(documents):
                    doc.metadata.update({
                        'source_file': pdf_path,
                        'paper_title': self.paper_titles[idx],
                        'paper_id': idx,
                        'page_number': doc_idx + 1,
                        'total_pages': len(documents)
                    })
                    all_documents.append(doc)
                
                print(f"   ✅ Loaded {len(documents)} pages from {pdf_path}")
                
            except Exception as e:
                print(f"   ❌ Error processing {pdf_path}: {e}")
                continue
        
        self.documents = all_documents
        print(f"\n📊 Total documents loaded: {len(all_documents)}")
        return all_documents

class LangChainTextSplitter:
    """
    Advanced text splitting using LangChain's text splitters.
    Implements multiple splitting strategies for optimal chunking.
    """
    
    def __init__(self):
        # Recursive Character Text Splitter for general use
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Token-based splitter for precise token control
        self.token_splitter = TokenTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        
        # Custom academic paper splitter
        self.academic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300,
            length_function=len,
            separators=[
                "\n## ",      # Section headers
                "\n### ",     # Subsection headers  
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentence endings
                " "           # Word boundaries
            ]
        )
    
    def split_documents(self, documents: List[Document], 
                       strategy: str = "recursive") -> List[Document]:
        """
        Split documents using specified strategy.
        
        Args:
            documents: List of LangChain documents
            strategy: "recursive", "token", or "academic"
        """
        print(f"🔪 Splitting {len(documents)} documents using {strategy} strategy...")
        
        if strategy == "recursive":
            splitter = self.recursive_splitter
        elif strategy == "token":
            splitter = self.token_splitter
        elif strategy == "academic":
            splitter = self.academic_splitter
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Split documents
        split_docs = splitter.split_documents(documents)
        
        # Add chunk metadata
        for idx, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': idx,
                'splitting_strategy': strategy,
                'chunk_size': len(doc.page_content),
                'word_count': len(doc.page_content.split())
            })
        
        print(f"   ✅ Created {len(split_docs)} chunks")
        print(f"   📊 Average chunk size: {np.mean([len(doc.page_content) for doc in split_docs]):.0f} characters")
        
        return split_docs

class LangChainVectorStore:
    """
    FAISS vector store implementation using LangChain with HuggingFace embeddings.
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model_name = embedding_model_name
        
        # Initialize HuggingFace embeddings
        print(f"🤗 Initializing HuggingFace embeddings: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        self.retriever = None
        
    def create_vector_store(self, documents: List[Document]) -> FAISS:
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
    
    def save_vector_store(self, path: str):
        """Save FAISS vector store to disk."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        self.vector_store.save_local(path)
        print(f"💾 Vector store saved to: {path}")
    
    def load_vector_store(self, path: str):
        """Load FAISS vector store from disk."""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            print(f"📁 Vector store loaded from: {path}")
            return self.vector_store
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return None
    
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

class LangChainHuggingFaceLLM:
    """
    HuggingFace LLM integration using LangChain's HuggingFacePipeline.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.llm_pipeline = None
        self.llm = None
        self.setup_model()
    
    def setup_model(self):
        """Setup HuggingFace model pipeline."""
        print(f"🤖 Setting up HuggingFace model: {self.model_name}")
        
        model_options = [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "distilgpt2", 
            "microsoft/DialoGPT-small"
        ]
        
        for model_name in model_options:
            try:
                print(f"   Attempting to load: {model_name}")
                
                # Create text generation pipeline
                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    device=0 if torch.cuda.is_available() else -1,
                    return_full_text=False
                )
                
                # Wrap in LangChain HuggingFacePipeline
                self.llm = HuggingFacePipeline(
                    pipeline=self.llm_pipeline,
                    model_kwargs={
                        "temperature": 0.7,
                        "max_length": 512,
                        "do_sample": True
                    }
                )
                
                self.model_name = model_name
                print(f"   ✅ Successfully loaded: {model_name}")
                break
                
            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
                continue
        
        if self.llm is None:
            print("⚠️  Could not load any HuggingFace model, using fallback")
            self.llm = self._create_fallback_llm()
    
    def _create_fallback_llm(self):
        """Create a simple fallback LLM."""
        from langchain.llms.base import LLM
        
        class FallbackLLM(LLM):
            def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
                # Simple template-based responses
                if "transformer" in prompt.lower() or "attention" in prompt.lower():
                    return "The Transformer architecture uses self-attention mechanisms to process sequences in parallel, making it highly efficient for natural language processing tasks."
                elif "bert" in prompt.lower():
                    return "BERT (Bidirectional Encoder Representations from Transformers) uses bidirectional training to understand context from both left and right directions."
                elif "gpt" in prompt.lower():
                    return "GPT models are autoregressive language models that generate text by predicting the next token based on previous context."
                else:
                    return "Based on the research papers, this topic involves advanced natural language processing techniques and transformer architectures."
            
            @property
            def _llm_type(self) -> str:
                return "fallback"
        
        return FallbackLLM()
    
    def get_llm(self):
        """Get the LangChain LLM instance."""
        return self.llm

class LangChainRAGChains:
    """
    RAG chains implementation using LangChain's RetrievalQA and ConversationalRetrievalChain.
    """
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
        # Custom prompt template for academic papers
        self.qa_template = """You are an AI assistant specialized in analyzing academic papers about natural language processing and transformer architectures.

Use the following pieces of context from research papers to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't have enough information.
Keep the answer concise but informative, and mention relevant paper names when possible.

Context:
{context}

Question: {question}

Answer: """
        
        self.QA_PROMPT = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )
        
        # Conversational prompt template
        self.conversational_template = """You are an AI assistant specialized in academic research papers on NLP and transformers.
        
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Take into account the conversation history and the context from research papers.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

        self.CONDENSE_QUESTION_PROMPT = PromptTemplate(
            template=self.conversational_template,
            input_variables=["chat_history", "question"]
        )
        
        # Initialize chains
        self.setup_chains()
    
    def setup_chains(self):
        """Setup RetrievalQA and Conversational chains."""
        print("⛓️  Setting up LangChain RAG chains...")
        
        # RetrievalQA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.QA_PROMPT},
            return_source_documents=True
        )
        
        # Conversational Memory (last 4 interactions)
        self.memory = ConversationBufferWindowMemory(
            k=4,  # Remember last 4 interactions
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Conversational Retrieval Chain
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
            qa_prompt=self.QA_PROMPT,
            return_source_documents=True
        )
        
        print("   ✅ Chains initialized successfully")
    
    def ask_question(self, question: str, use_conversation: bool = False) -> Dict[str, Any]:
        """Ask a question using RAG chains."""
        if use_conversation:
            # Use conversational chain with memory
            result = self.conversational_chain({"question": question})
        else:
            # Use simple QA chain
            result = self.qa_chain({"query": question})
        
        return result
    
    def get_memory_statistics(self) -> Dict:
        """Get conversation memory statistics."""
        if hasattr(self.memory, 'buffer'):
            messages = self.memory.buffer
            return {
                'total_messages': len(messages),
                'memory_utilization': len(messages) / 8,  # 4 interactions = 8 messages (Q&A pairs)
                'last_interaction': messages[-1].content if messages else None
            }
        return {'total_messages': 0, 'memory_utilization': 0}
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()

class LangChainRAGEvaluator:
    """
    Comprehensive evaluation framework for LangChain RAG system.
    """
    
    def __init__(self):
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        
        # Evaluation weights
        self.weights = {
            'context_relevance': 0.25,
            'answer_faithfulness': 0.25,
            'answer_relevance': 0.25,
            'context_recall': 0.15,
            'response_quality': 0.10
        }
    
    def evaluate_context_relevance(self, question: str, contexts: List[str]) -> float:
        """Evaluate relevance of retrieved contexts to the question."""
        if not contexts:
            return 0.0
        
        question_embedding = self.sentence_model.encode([question])
        context_embeddings = self.sentence_model.encode(contexts)
        
        similarities = util.cos_sim(question_embedding, context_embeddings)[0]
        
        # Weighted average with exponential decay
        weights = np.exp(-np.arange(len(similarities)) * 0.5)
        weights = weights / weights.sum()
        
        relevance_score = float(np.sum(similarities.cpu().numpy() * weights))
        return max(0, min(1, relevance_score))
    
    def evaluate_answer_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Evaluate faithfulness of answer to contexts."""
        if not contexts or not answer:
            return 0.0
        
        combined_context = " ".join(contexts)
        
        # Semantic similarity
        answer_embedding = self.sentence_model.encode([answer])
        context_embedding = self.sentence_model.encode([combined_context])
        semantic_sim = float(util.cos_sim(answer_embedding, context_embedding)[0][0])
        
        # Token overlap
        answer_tokens = set(answer.lower().split())
        context_tokens = set(combined_context.lower().split())
        token_overlap = len(answer_tokens.intersection(context_tokens)) / max(1, len(answer_tokens))
        
        faithfulness = 0.7 * semantic_sim + 0.3 * token_overlap
        return max(0, min(1, faithfulness))
    
    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """Evaluate relevance of answer to question."""
        if not question or not answer:
            return 0.0
        
        question_embedding = self.sentence_model.encode([question])
        answer_embedding = self.sentence_model.encode([answer])
        
        relevance = float(util.cos_sim(question_embedding, answer_embedding)[0][0])
        return max(0, min(1, relevance))
    
    def evaluate_context_recall(self, question: str, contexts: List[str]) -> float:
        """Evaluate how well contexts cover the question."""
        if not contexts:
            return 0.0
        
        question_tokens = set(question.lower().split())
        all_context_tokens = set(" ".join(contexts).lower().split())
        
        coverage = len(question_tokens.intersection(all_context_tokens)) / max(1, len(question_tokens))
        
        # Context diversity
        if len(contexts) > 1:
            context_embeddings = self.sentence_model.encode(contexts)
            similarities = util.cos_sim(context_embeddings, context_embeddings)
            avg_sim = (similarities.sum() - len(contexts)) / (len(contexts) * (len(contexts) - 1))
            diversity = 1 - float(avg_sim)
        else:
            diversity = 0.5
        
        recall = 0.7 * coverage + 0.3 * diversity
        return max(0, min(1, recall))
    
    def evaluate_response_quality(self, answer: str) -> float:
        """Evaluate overall response quality."""
        if not answer:
            return 0.0
        
        # Readability
        try:
            readability = max(0, min(1, textstat.flesch_reading_ease(answer) / 100.0))
        except:
            readability = 0.5
        
        # Length appropriateness
        word_count = len(answer.split())
        if 30 <= word_count <= 150:
            length_score = 1.0
        elif word_count < 30:
            length_score = word_count / 30.0
        else:
            length_score = max(0.3, 150.0 / word_count)
        
        # Coherence (simple sentence count)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        coherence = min(1.0, len(sentences) / 3.0)
        
        quality = 0.4 * readability + 0.3 * length_score + 0.3 * coherence
        return max(0, min(1, quality))
    
    def comprehensive_evaluation(self, question: str, answer: str, 
                               source_documents: List[Document]) -> Dict[str, float]:
        """Perform comprehensive evaluation."""
        contexts = [doc.page_content for doc in source_documents] if source_documents else []
        
        metrics = {
            'context_relevance': self.evaluate_context_relevance(question, contexts),
            'answer_faithfulness': self.evaluate_answer_faithfulness(answer, contexts),
            'answer_relevance': self.evaluate_answer_relevance(question, answer),
            'context_recall': self.evaluate_context_recall(question, contexts),
            'response_quality': self.evaluate_response_quality(answer)
        }
        
        # Calculate weighted overall score
        overall_score = sum(metrics[key] * self.weights[key] for key in metrics)
        metrics['overall_score'] = overall_score
        
        return metrics

class LangChainRAGApplication:
    """
    Main RAG application using LangChain components.
    """
    
    def __init__(self):
        self.pdf_ingestion = LangChainPDFIngestion()
        self.text_splitter = LangChainTextSplitter()
        self.vector_store = LangChainVectorStore()
        self.llm_integration = LangChainHuggingFaceLLM()
        self.rag_chains = None
        self.evaluator = LangChainRAGEvaluator()
        self.interaction_history = []
        
        # Test questions for evaluation
        self.test_questions = [
            "What is the key innovation introduced in the 'Attention Is All You Need' paper?",
            "How does BERT's bidirectional training differ from traditional language models?", 
            "What are the main architectural components of the Transformer model?",
            "How does GPT-3 achieve few-shot learning capabilities?",
            "What is the difference between encoder-only and decoder-only transformer architectures?",
            "How does RoBERTa improve upon the original BERT model?",
            "What is the T5 text-to-text transfer transformer approach?",
            "How do attention mechanisms solve the vanishing gradient problem in RNNs?",
            "What are the advantages of self-attention over recurrent and convolutional layers?",
            "How do these transformer-based models handle long-range dependencies in text?"
        ]
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system."""
        print("🚀 Initializing LangChain RAG System")
        print("=" * 60)
        
        try:
            # Step 1: Load and process PDFs
            print("\n📚 Step 1: PDF Loading and Processing")
            documents = self.pdf_ingestion.load_and_process_documents()
            
            if not documents:
                print("❌ No documents loaded!")
                return False
            
            # Step 2: Text splitting
            print("\n🔪 Step 2: Text Splitting")
            split_documents = self.text_splitter.split_documents(documents, strategy="academic")
            
            # Step 3: Create vector store
            print("\n🗂️  Step 3: Vector Store Creation")
            vector_store = self.vector_store.create_vector_store(split_documents)
            
            # Step 4: Setup RAG chains
            print("\n⛓️  Step 4: RAG Chains Setup")
            llm = self.llm_integration.get_llm()
            self.rag_chains = LangChainRAGChains(llm, self.vector_store.retriever)
            
            # Save vector store
            self.vector_store.save_vector_store("langchain_faiss_index")
            
            print("\n✅ LangChain RAG System Initialized Successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_question(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Process a question through the RAG system."""
        if self.rag_chains is None:
            raise ValueError("RAG system not initialized!")
        
        print(f"🤔 Processing: {question[:60]}...")
        
        # Get answer using RAG chains
        result = self.rag_chains.ask_question(question, use_conversation=use_memory)
        
        # Extract components
        answer = result.get('answer') or result.get('result', '')
        source_docs = result.get('source_documents', [])
        
        # Evaluate response
        evaluation = self.evaluator.comprehensive_evaluation(question, answer, source_docs)
        
        # Compile response
        response = {
            'question': question,
            'answer': answer,
            'source_documents': source_docs,
            'evaluation_metrics': evaluation,
            'memory_stats': self.rag_chains.get_memory_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.interaction_history.append(response)
        
        print(f"   ✅ Overall Score: {evaluation['overall_score']:.3f}")
        return response
    
    def run_evaluation_suite(self) -> Dict[str, Any]:
        """Run complete evaluation with test questions."""
        print("\n🔍 Running Evaluation Suite")
        print("=" * 60)
        
        results = []
        all_metrics = []
        
        for i, question in enumerate(self.test_questions):
            print(f"\nQuestion {i+1}/{len(self.test_questions)}:")
            
            response = self.process_question(question, use_memory=True)
            results.append(response)
            all_metrics.append(response['evaluation_metrics'])
        
        # Calculate aggregate statistics
        metric_keys = ['context_relevance', 'answer_faithfulness', 'answer_relevance', 
                       'context_recall', 'response_quality', 'overall_score']
        
        aggregate_metrics = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregate_metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        evaluation_summary = {
            'test_questions': self.test_questions,
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics,
            'system_performance': {
                'overall_mean_score': aggregate_metrics['overall_score']['mean'],
                'total_questions': len(self.test_questions),
                'memory_stats': self.rag_chains.get_memory_statistics()
            },
            'system_config': {
                'embedding_model': self.vector_store.embedding_model_name,
                'llm_model': self.llm_integration.model_name,
                'text_splitter': 'RecursiveCharacterTextSplitter',
                'vector_store': 'FAISS',
                'memory_size': 4
            },
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Evaluation Complete!")
        print(f"📊 Overall Performance: {aggregate_metrics['overall_score']['mean']:.3f} ± {aggregate_metrics['overall_score']['std']:.