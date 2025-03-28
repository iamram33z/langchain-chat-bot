import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from huggingface_hub import InferenceClient
import time
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re
import json

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Constants
PDF_PATH = os.path.join(os.getcwd(), "chatbot", "data", "fintech_guide.pdf")
LLM_MODEL = "meta-llama/Llama-3.2-1B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIES = 3
TIMEOUT = 30
RETRIEVAL_K = 3
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
NEWS_API_URL = "https://newsapi.org/v2/everything"


class FintechRAGPipeline:
    def __init__(self):
        """Initialize Fintech RAG pipeline components"""
        self.retriever = None
        self.vectorstore = None
        self.embeddings = None
        self.chunks = None
        self.documents = None
        self.llm_client = None
        self.rag_chain = None
        self.initialize_components()

    def initialize_components(self):
        """Initialize all components with error handling"""
        print("🚀 Initializing Fintech RAG pipeline...")
        start = time.time()

        try:
            # 1. Load and process documents
            self.load_documents()

            # 2. Initialize embeddings and vector store
            self.initialize_embeddings()

            # 3. Initialize LLM client
            self.initialize_llm_client()

            # 4. Build the RAG chain
            self.build_rag_chain()

            print(f"✅ Initialization completed in {time.time() - start:.2f}s")

        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def load_documents(self):
        """Load and process Fintech documents"""
        print("📂 Loading Fintech documents...")

        # Load PDF documents
        pdf_docs = PyPDFLoader(PDF_PATH).load()
        self.documents = pdf_docs

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Loaded {len(self.chunks)} chunks from {len(self.documents)} documents")

    def initialize_embeddings(self):
        """Initialize embeddings and vector store"""
        print("🔍 Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )

    def initialize_llm_client(self):
        """Initialize the LLM client with configuration"""
        print("💡 Initializing LLM client...")
        self.llm_client = InferenceClient(
            model=LLM_MODEL,
            token=HUGGINGFACE_API_KEY,
            timeout=TIMEOUT
        )

    def build_rag_chain(self):
        """Build the Fintech RAG chain with dynamic news embedding"""
        print("⛓️ Building Fintech RAG chain...")

        prompt_template = """
        You are a Fintech expert assistant. Answer the question based on the context.
        For financial news, analyze trends and implications.

        Context: {context}

        Question: {question}

        Provide a comprehensive response:
        1. Start with a direct answer
        2. Include relevant financial concepts
        3. Add analysis of recent trends if available
        4. Highlight key insights
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_prompt(prompt_value):
            return prompt_value.to_string()

        self.rag_chain = (
                RunnableParallel({
                    "context": RunnableLambda(self.retrieve_with_news),
                    "question": RunnablePassthrough()
                })
                | prompt
                | RunnableLambda(format_prompt)
                | RunnableLambda(self.generate_response_with_retry)
        )

    def extract_keywords(self, question: str) -> List[str]:
        """Extract quoted financial keywords from question"""
        return re.findall(r'"([^"]*)"', question)

    def search_news(self, keywords: List[str]) -> List[Document]:
        """Search financial news and return as Documents"""
        if not keywords or not NEWS_API_KEY:
            return []

        news_docs = []
        for keyword in keywords[:3]:  # Limit to 3 keywords
            try:
                params = {
                    'q': f"{keyword} AND (finance OR fintech OR banking)",
                    'apiKey': NEWS_API_KEY,
                    'pageSize': 5,
                    'sortBy': 'publishedAt',
                    'language': 'en'
                }

                response = requests.get(NEWS_API_URL, params=params, timeout=TIMEOUT)
                response.raise_for_status()

                articles = response.json().get('articles', [])
                for article in articles[:5]:
                    content = (
                        f"Financial News: {article.get('title', 'No title')}\n"
                        f"Source: {article.get('source', {}).get('name', 'Unknown')}\n"
                        f"Date: {article.get('publishedAt', 'Unknown date')}\n"
                        f"Summary: {article.get('description', 'No description')}\n"
                    )
                    metadata = {
                        "source": "news_api",
                        "title": article.get('title', ''),
                        "url": article.get('url', ''),
                        "date": article.get('publishedAt', ''),
                        "keyword": keyword
                    }
                    news_docs.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"⚠️ News API error for keyword '{keyword}': {str(e)}")
                continue

        return news_docs

    def retrieve_with_news(self, question: str) -> str:
        """Retrieve context with temporary news embeddings"""
        # Get regular document context
        docs = self.retriever.invoke(question)

        # Check for news keywords
        keywords = self.extract_keywords(question)
        if keywords:
            news_docs = self.search_news(keywords)
            if news_docs:
                # Create temporary vectorstore with news
                temp_store = FAISS.from_documents(news_docs, self.embeddings)
                # Add news results to context
                news_docs = temp_store.similarity_search(question, k=2)
                docs.extend(news_docs)

        return "\n\n---\n\n".join(
            f"Source: {doc.metadata.get('source', 'fintech_guide')}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        )

    def generate_response_with_retry(self, prompt_text: str) -> str:
        """Generate response with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_client.text_generation(
                    prompt_text,
                    max_new_tokens=600,
                    temperature=0.6,
                    do_sample=True,
                    return_full_text=False
                )
                return self.process_llm_response(response)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    wait = 2 ** attempt
                    print(f"⚠️ Service unavailable, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return self.get_fallback_response(prompt_text)
                time.sleep(1)

    def process_llm_response(self, response) -> str:
        """Process and clean the LLM response"""
        if isinstance(response, str):
            return response.strip()
        elif hasattr(response, "generated_text"):
            return response.generated_text.strip()
        return str(response).strip()

    def get_fallback_response(self, question: str) -> str:
        """Provide fallback response when all retries fail"""
        try:
            docs = self.retriever.invoke(question)
            if docs:
                return (
                    "I found some relevant Fintech information:\n\n"
                    f"{docs[0].page_content[:400]}...\n\n"
                    "Please refine your question if you need more details."
                )
        except Exception as e:
            print(f"⚠️ Fallback failed: {str(e)}")

        return ("I'm experiencing technical difficulties. "
                "Please try again later or rephrase your question.")

    def get_response(self, question: str) -> str:
        """Get RAG response with timing and error handling"""
        start_time = time.time()
        try:
            response = self.rag_chain.invoke(question)
            print(f"⏱️ Response generated in {time.time() - start_time:.2f}s")
            return response
        except Exception as e:
            print(f"⚠️ Error in get_response: {str(e)}")
            return self.get_fallback_response(question)


# Initialize pipeline globally
fintech_rag_pipeline = FintechRAGPipeline()


def get_rag_response(question: str) -> str:
    """Public interface to get RAG response"""
    return fintech_rag_pipeline.get_response(question)