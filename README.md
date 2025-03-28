# RAG-Powered Chatbot with FastAPI
A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your PDF documents, built with FastAPI, LangChain, and HuggingFace models.

## Features
- **Document Intelligence**: Extract knowledge from PDF files
- **Semantic Search**: Find relevant document passages using HuggingFace embeddings
- **LLM Responses**: Generate answers using Meta's Llama 3 model
- **Web Interface**: User-friendly chat UI with response timing
- **Production-Ready**: FastAPI backend with error handling and logging

## Technology Stack
| Component               | Technology                          |
|-------------------------|-------------------------------------|
| Backend Framework       | FastAPI                             |
| Frontend                | Jinja2 Templates + Static HTML/CSS  |
| RAG Pipeline            | LangChain                           |
| Embeddings              | HuggingFace Sentence Transformers   |
| LLM                     | Meta Llama 3 (via HuggingFace Hub)  |
| Vector Store            | FAISS                               |
| PDF Processing          | PyPDFLoader                         |

## Installation
```bash
pip install -r requirements.txt
```

### Prerequisites

- Python 3.9+
- Poetry (recommended) or pip
- HuggingFace API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/iamram33z/langchain-chat-bot.git
   cd rag-chatbot


### To Load the Application
```bash
uvicorn chatbot.app.main:app --reload

