import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

class FAISSVectorStore:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.dimension = 384  # all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Добавляет документы в векторное хранилище"""
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Поиск похожих документов"""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(distance)
                })
        return results
    
    def save(self, path: str):
        """Сохранение индекса"""
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.metadata}, f)
    
    def load(self, path: str):
        """Загрузка индекса"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/documents.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
