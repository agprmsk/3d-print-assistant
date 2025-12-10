from fastapi import FastAPI
from pydantic import BaseModel
from .rag_pipeline import handle_user_query

app = FastAPI(title="3D Printing Assistant")

class ChatRequest(BaseModel):
    query: str
    dialog_context: str | None = ""

class ChatResponse(BaseModel):
    category: str
    answer: str
    sources: list[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    result = handle_user_query(req.query, dialog_context=req.dialog_context or "")
    return ChatResponse(**result)