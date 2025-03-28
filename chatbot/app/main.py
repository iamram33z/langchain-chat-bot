from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

from chatbot.app.rag_pipeline import NEWS_API_KEY, get_rag_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fintech AI Assistant",
    version="2.1.0",
    description="API for Fintech knowledge and financial news"
)

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

# Validate paths
if not TEMPLATE_DIR.exists():
    raise RuntimeError(f"Template directory not found: {TEMPLATE_DIR}")
if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

logger.info(f"Template directory: {TEMPLATE_DIR}")
logger.info(f"Static directory: {STATIC_DIR}")

# Initialize templates and static files
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up Fintech AI Assistant")
    try:
        test_response = get_rag_response("Explain \"blockchain\" technology")
        logger.info("Fintech RAG pipeline warm-up successful")
    except Exception as e:
        logger.error(f"Fintech RAG pipeline warm-up failed: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def chat_interface(
        request: Request,
        error: Optional[str] = None
):
    """Render the main chat interface"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": [],
            "error": error,
            "current_year": datetime.now().year
        }
    )


@app.post("/ask", response_class=HTMLResponse)
async def handle_question(
        request: Request,
        question: str = Form(..., min_length=1, max_length=500),
        chat_history: List[Dict] = Form([])
) -> HTMLResponse:
    """Handle user questions with Fintech RAG"""
    start_time = time.time()
    logger.info(f"Processing Fintech question: {question[:100]}...")

    try:
        # Get RAG response with timing
        response_start = time.time()
        response = get_rag_response(question)
        response_time = time.time() - response_start

        # Create new chat entry
        new_entry = {
            "user": question,
            "bot": response,
            "time": f"{response_time:.1f}s",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Update chat history (keep last 10 messages)
        updated_history = (chat_history + [new_entry])[-10:]

        logger.info(f"Successfully processed question in {time.time() - start_time:.2f}s")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "chat_history": updated_history,
                "success": True
            }
        )

    except HTTPException as http_error:
        logger.error(f"HTTP error processing question: {http_error.detail}")
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Service unavailable. Please try again later.",
                "chat_history": chat_history
            },
            status_code=http_error.status_code
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "We're experiencing technical difficulties. Please try again.",
                "chat_history": chat_history
            },
            status_code=500
        )


@app.get("/api/chat", response_class=JSONResponse)
async def api_chat_endpoint(question: str):
    """API endpoint for programmatic access"""
    try:
        response = get_rag_response(question)
        return {
            "response": response,
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "rag_pipeline": "operational",
            "news_api": "active" if NEWS_API_KEY else "inactive",
            "llm_connection": "active"
        },
        "version": app.version
    }
    return JSONResponse(content=health_info)


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return templates.TemplateResponse(
        "404.html",
        {"request": request},
        status_code=404
    )


# Store startup time
app.startup_time = time.time()