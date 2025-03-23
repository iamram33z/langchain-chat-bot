from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from chatbot.app.rag_pipeline import get_rag_response  # Import RAG function

app = FastAPI()

# Load templates from the 'templates' folder
templates = Jinja2Templates(directory="chatbot/templates")

@app.get("/", response_class=HTMLResponse)
async def serve_chat_page(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": []})

@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    """Process user input and get RAG response."""
    response = get_rag_response(question)  # Call the RAG pipeline
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": [{"user": question, "bot": response}]})