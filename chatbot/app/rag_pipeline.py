import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from huggingface_hub import InferenceClient
import time
from typing import Optional
import requests

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Constants
PDF_PATH = os.path.join(os.getcwd(), "chatbot", "data", "codeprolk.pdf")
LLM_MODEL = "meta-llama/Llama-3.2-1B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RETRIES = 3
TIMEOUT = 30


class RAGPipeline:
    def __init__(self):
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
        print("Initializing RAG pipeline...")
        start = time.time()

        try:
            # 1. Load and process PDF
            self.load_pdf()

            # 2. Initialize embeddings and vector store
            self.initialize_embeddings()

            # 3. Initialize LLM client
            self.llm_client = InferenceClient(
                model=LLM_MODEL,
                token=HUGGINGFACE_API_KEY,
                timeout=TIMEOUT
            )

            # 4. Build the RAG chain
            self.build_rag_chain()

            print(f"Initialization completed in {time.time() - start:.2f}s")

        except Exception as e:
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def load_pdf(self):
        """Load and process PDF document"""
        print("Loading PDF...")
        loader = PyPDFLoader(PDF_PATH)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Loaded {len(self.chunks)} chunks from PDF")

    def initialize_embeddings(self):
        """Initialize embeddings and vector store"""
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def build_rag_chain(self):
        """Build the RAG chain using LangChain's Runnable composition"""
        # Define prompt template
        prompt_template = """
        Answer the question based on the context below.

        Context: {context}

        Question: {question}

        Provide a concise answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Define the chain components
        retrieve_context = RunnableLambda(self.retrieve_context)
        generate_response = RunnableLambda(self.generate_response)

        # Build the chain with error handling wrapper
        self.rag_chain = (
                RunnableParallel({
                    "context": retrieve_context,
                    "question": RunnablePassthrough()
                })
                | prompt
                | generate_response
        )

    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from vector store"""
        docs = self.retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, input_dict: dict) -> str:
        """Generate response from LLM with proper formatting and error handling"""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm_client.text_generation(
                    input_dict["text"],  # The prompt template fills this
                    max_new_tokens=250,
                    temperature=0.5,
                    do_sample=True
                )

                # Handle different response formats
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, "generated_text"):
                    return response.generated_text.strip()
                return str(response).strip()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503:
                    wait = 2 ** attempt  # Exponential backoff
                    print(f"Service unavailable, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    return self.get_fallback_response(input_dict["question"])
                time.sleep(1)

    def get_fallback_response(self, question: str) -> str:
        """Fallback when all retries fail"""
        try:
            docs = self.retriever.invoke(question)
            if docs:
                return f"Here's relevant info: {docs[0].page_content[:300]}..."
        except:
            pass
        return "I'm having trouble answering right now. Please try again later."

    def get_response(self, question: str) -> str:
        """Main method to get RAG response"""
        return self.rag_chain.invoke(question)


# Initialize pipeline globally
rag_pipeline = RAGPipeline()


def get_rag_response(question: str) -> str:
    """Public interface to get RAG response"""
    return rag_pipeline.get_response(question)