"""Конфигурация проекта и переменные окружения"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed.jsonl"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Создание директорий
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Perplexity API
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PPLX_MODEL_GENERAL = os.getenv("PPLX_MODEL_GENERAL", "llama-3.1-sonar-small-128k-online")
PPLX_MODEL_STRICT = os.getenv("PPLX_MODEL_STRICT", "llama-3.1-sonar-small-128k-online")

# Параметры RAG
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K_DOCUMENTS = 6

# Embedding модель
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Проверка ключа API
if not PERPLEXITY_API_KEY:
    print("⚠️ ВНИМАНИЕ: PERPLEXITY_API_KEY не установлен в .env файле")
