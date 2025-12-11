"""
Упрощённая версия с FAISS вместо ChromaDB
"""
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class EmbeddingsStoreFAISS:
    """Векторное хранилище на базе FAISS"""
    
    def __init__(self, db_path: str = "data/faiss_index"):
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        
        print("📦 Загружаем модель эмбеддингов...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.dimension = 384
        
        self.index = None
        self.documents = []
        self.metadatas = []
    
    def build_from_articles(self, articles_path: str):
        """Строит индекс из статей"""
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"📄 Загружено {len(articles)} статей")
        
        if articles:
            print(f"🔍 Ключи первой статьи: {list(articles[0].keys())}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        chunks = []
        for article in articles:
            text = article.get('text') or article.get('content') or article.get('body') or ""
            
            if not text:
                print(f"⚠️ Статья без текста: {article.get('title', 'N/A')}")
                continue
            
            title = article.get('title', 'Без названия')
            url = article.get('url', '')
            category = article.get('category', 'unknown')
            
            texts = splitter.split_text(text)
            for chunk_text in texts:
                chunks.append({
                    "text": chunk_text,
                    "title": title,
                    "url": url,
                    "category": category
                })
        
        print(f"✂️ Создано {len(chunks)} чанков")
        
        if not chunks:
            print("❌ Нет данных для индексации!")
            return
        
        print("🔄 Создаём эмбеддинги...")
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        self.documents = texts
        self.metadatas = [
            {"title": c['title'], "url": c['url'], "category": c['category']}
            for c in chunks
        ]
        
        self.save()
        print(f"✅ Индекс создан: {len(texts)} чанков")
    
    def save(self):
        """Сохраняет индекс на диск"""
        faiss.write_index(self.index, os.path.join(self.db_path, "index.faiss"))
        with open(os.path.join(self.db_path, "documents.pkl"), 'wb') as f:
            pickle.dump((self.documents, self.metadatas), f)
    
    def load(self):
        """Загружает индекс с диска"""
        index_path = os.path.join(self.db_path, "index.faiss")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents, self.metadatas = pickle.load(f)
            return True
        return False
    
    def search(self, query: str, k: int = 3):
        """Поиск релевантных чанков"""
        query_embedding = self.model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        results = []
        for idx in indices[0]:
            results.append({
                "text": self.documents[idx],
                "metadata": self.metadatas[idx]
            })
        return results


if __name__ == "__main__":
    store = EmbeddingsStoreFAISS()
    
    articles_path = "data/raw/3dtoday_articles.json"
    if not os.path.exists(articles_path):
        print(f"❌ Файл {articles_path} не найден")
    else:
        store.build_from_articles(articles_path)
        
        print("\n🔍 Тестовый поиск:")
        results = store.search("Как печатать PLA пластиком?", k=2)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['metadata']['title']}")
            print(f"   {r['text'][:200]}...")
