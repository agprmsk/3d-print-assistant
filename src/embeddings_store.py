"""
Управление эмбеддингами через FAISS
"""
from app.vectorstore import FAISSVectorStore
from typing import List, Dict

class EmbeddingsStore:
    def __init__(self, index_path: str = "data/faiss_index/"):
        self.vectorstore = FAISSVectorStore()
        self.index_path = index_path
        
        # Загрузка существующего индекса
        try:
            self.vectorstore.load(index_path)
            print(f"✅ Индекс загружен из {index_path}")
        except:
            print(f"⚠️ Индекс не найден в {index_path}")
    
    def add(self, texts: List[str], metadatas: List[Dict] = None):
        """Добавляет тексты в хранилище"""
        self.vectorstore.add_documents(texts, metadatas)
        self.vectorstore.save(self.index_path)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск похожих документов"""
        return self.vectorstore.search(query, k)
    
    def save(self):
        """Сохраняет индекс"""
        self.vectorstore.save(self.index_path)
