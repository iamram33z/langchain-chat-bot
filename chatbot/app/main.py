from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import time
import logging
from typing import List, Dict

from chatbot.app.rag_pipeline import get_rag_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

# Debug paths
logger.info(f"Template directory: {TEMPLATE_DIR}")
logger.info(f"Static directory: {STATIC_DIR}")

# Initialize templates and static files
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Type aliases for better code documentation
ChatMessage = Dict[str, str]
ChatHistory = List[ChatMessage]


@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Render the main chat interface"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": []}
    )


@app.post("/ask", response_class=HTMLResponse)
async def handle_question(
        request: Request,
        question: str = Form(..., min_length=1, max_length=500)
) -> HTMLResponse:
    """
    Handle user questions and return RAG-generated responses

    Args:
        request: The incoming request
        question: The user's question (form input)

    Returns:
        Rendered HTML response with chat history
    """
    start_time = time.time()
    logger.info(f"Processing question: {question[:50]}...")

    try:
        # Get RAG response
        response = get_rag_response(question)
        elapsed = time.time() - start_time
        logger.info(f"Response generated in {elapsed:.2f}s")

        # Create chat history entry
        chat_entry = {
            "user": question,
            "bot": response,
            "time": f"{elapsed:.1f}s"
        }

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "chat_history": [chat_entry]
            }
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        error_msg = "Sorry, I encountered an error processing your question. Please try again."

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": error_msg,
                "chat_history": []
            },
            status_code=500 if isinstance(e, HTTPException) else 200
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}