from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import time

from chatbot.app.rag_pipeline import get_rag_response

app = FastAPI()

# Configure paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR  / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

# Debug paths
print(f"Template dir: {TEMPLATE_DIR}")
print(f"Static dir: {STATIC_DIR}")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": []}
    )

@app.post("/ask", response_class=HTMLResponse)
async def handle_question(request: Request, question: str = Form(...)):
    start_time = time.time()
    
    try:
        response = get_rag_response(question)
        elapsed = time.time() - start_time
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "chat_history": [{
                    "user": question,
                    "bot": response,
                    "time": f"{elapsed:.1f}s"
                }]
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Error: {str(e)}",
                "chat_history": []
            }
        )