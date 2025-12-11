# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Загрузка переменных окружения
load_dotenv()

# Инициализация FastAPI
app = FastAPI(
    title="3D Print Assistant API",
    description="RAG-система для помощи в 3D-печати",
    version="1.0.0"
)

# Инициализация RAG при старте приложения
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Инициализация RAG-системы при запуске"""
    global rag_pipeline
    print("🚀 Запуск 3D Print Assistant API...")
    print("🔧 Инициализация RAG-системы...")
    rag_pipeline = RAGPipeline()
    print("✅ RAG-система готова к работе!")

# Модели данных
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources_count: int

# Эндпоинты
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "3D Print Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "/query": "POST - Задать вопрос системе",
            "/health": "GET - Проверка состояния",
            "/docs": "GET - Документация API"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка состояния системы"""
    return {
        "status": "healthy",
        "rag_initialized": rag_pipeline is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Задать вопрос RAG-системе
    
    Args:
        request: Объект запроса с вопросом и параметрами
    
    Returns:
        Ответ системы с источниками
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-система не инициализирована"
        )
    
    try:
        # Получение ответа
        answer = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k
        )
        
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources_count=request.top_k
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке запроса: {str(e)}"
        )

# Запуск: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
