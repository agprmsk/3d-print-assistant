"""
RAG Pipeline - полная версия с FAISS и Perplexity API
"""
import os
import sys
import json
from typing import Optional, Dict, Any, List, Literal
from pathlib import Path

# Добавляем корневую директорию в путь для импортов
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Импорты с обработкой ошибок для прямого запуска
try:
    from .config import DATA_DIR, TOP_K_DOCUMENTS
    from .models import pplx_chat
except ImportError:
    from src.config import DATA_DIR, TOP_K_DOCUMENTS
    from src.models import pplx_chat


# Типы категорий
Category = Literal[
    "основы",
    "подбор_материала",
    "настройка_принтера",
    "диагностика_дефектов",
    "слайсер",
    "другое"
]


class RAGPipeline:
    """
    Полная RAG-система с несколькими агентами:
    1. Классификатор - определяет категорию запроса
    2. Поисковик - ищет релевантные документы
    3. Консультант - формирует детальный ответ
    4. Валидатор - проверяет безопасность рекомендаций
    """
    
    def __init__(self):
        """Инициализация RAG-системы"""
        self.knowledge_base = []
        self.faiss_index = None
        self.embeddings_model = None
        self._load_knowledge_base()
        self._load_faiss_index()
    
    def _load_knowledge_base(self):
        """Загрузка базы знаний"""
        kb_path = DATA_DIR / "processed.jsonl"
        
        if not kb_path.exists():
            print(f"⚠️ База знаний не найдена: {kb_path}")
            return
        
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.knowledge_base.append(json.loads(line))
            print(f"✅ Загружено {len(self.knowledge_base)} документов")
        except Exception as e:
            print(f"❌ Ошибка загрузки базы знаний: {e}")
    
    def _load_faiss_index(self):
        """Загрузка FAISS индекса"""
        faiss_path = DATA_DIR / "faiss_index"
        index_file = faiss_path / "index.faiss"
        
        if not index_file.exists():
            print(f"⚠️ FAISS индекс не найден: {index_file}")
            print("Запустите: python -m src.embeddings_store_faiss")
            return
        
        try:
            import faiss
            self.faiss_index = faiss.read_index(str(index_file))
            print(f"✅ FAISS индекс загружен ({self.faiss_index.ntotal} векторов)")
            
            from sentence_transformers import SentenceTransformer
            self.embeddings_model = SentenceTransformer('intfloat/multilingual-e5-large')
            print("✅ Модель эмбеддингов загружена")
            
        except ImportError:
            print("❌ Установите: pip install faiss-cpu sentence-transformers")
        except Exception as e:
            print(f"❌ Ошибка загрузки FAISS: {e}")
    
    def _classify_query(self, user_query: str) -> Category:
        """Классификация запроса пользователя"""
        system_prompt = (
            "Ты классификатор запросов по 3D-печати. "
            "Определи категорию запроса и верни ОДНО слово из списка:\n"
            "- основы\n"
            "- подбор_материала\n"
            "- настройка_принтера\n"
            "- диагностика_дефектов\n"
            "- слайсер\n"
            "- другое\n\n"
            "Отвечай ТОЛЬКО одним словом из списка."
        )
        
        try:
            response = pplx_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.0,
                max_tokens=10
            )
            
            response_lower = response.lower().strip()
            for category in ["основы", "подбор_материала", "настройка_принтера", 
                           "диагностика_дефектов", "слайсер"]:
                if category in response_lower:
                    return category  # type: ignore
            
            return "другое"  # type: ignore
            
        except Exception as e:
            print(f"⚠️ Ошибка классификации: {e}")
            return "другое"  # type: ignore
    
    def _search_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Поиск релевантных документов через FAISS"""
        if not self.faiss_index or not self.embeddings_model:
            return self._simple_text_search(query, top_k)
        
        try:
            query_embedding = self.embeddings_model.encode([query])[0]
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            distances, indices = self.faiss_index.search(query_vector, top_k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.knowledge_base):
                    doc = self.knowledge_base[idx].copy()
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"⚠️ Ошибка FAISS поиска: {e}")
            return self._simple_text_search(query, top_k)
    
    def _simple_text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Упрощенный текстовый поиск"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.knowledge_base:
            score = 0
            content = doc.get('content', '').lower()
            title = doc.get('title', '').lower()
            
            for word in query_lower.split():
                if len(word) > 2:
                    score += content.count(word)
                    score += title.count(word) * 3
            
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _generate_answer(
        self, 
        user_query: str, 
        category: Category,
        documents: List[Dict[str, Any]],
        dialog_context: str = ""
    ) -> str:
        """Генерация детального ответа"""
        if not documents:
            return self._generate_fallback_answer(user_query, category)
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Без заголовка')
            content = doc.get('content', '')[:800]
            url = doc.get('source_url', 'N/A')
            
            context_parts.append(
                f"[Документ {i}]\n"
                f"Заголовок: {title}\n"
                f"Источник: {url}\n"
                f"Содержание: {content}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        system_prompt = (
            "Ты эксперт по 3D-печати с многолетним опытом. "
            "Твоя задача - дать максимально полезный и структурированный ответ.\n\n"
            "ВАЖНО:\n"
            "- Используй ТОЛЬКО факты из предоставленного контекста\n"
            "- Если информации недостаточно - честно скажи об этом\n"
            "- Давай конкретные параметры (температуры, скорости, и т.д.)\n"
            "- Структурируй ответ по пунктам"
        )
        
        user_prompt = (
            f"Категория: {category}\n"
            f"Контекст: {dialog_context or 'Нет'}\n\n"
            f"База знаний:\n{context_text}\n\n"
            f"Вопрос: {user_query}\n\n"
            "Дай структурированный ответ."
        )
        
        try:
            answer = pplx_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=1200
            )
            
            answer += "\n\n📚 Источники:\n"
            for i, doc in enumerate(documents, 1):
                title = doc.get('title', 'Без заголовка')
                url = doc.get('source_url', 'N/A')
                answer += f"{i}. {title} - {url}\n"
            
            return answer
            
        except Exception as e:
            print(f"⚠️ Ошибка генерации ответа: {e}")
            return self._generate_fallback_answer(user_query, category)
    
    def _generate_fallback_answer(self, query: str, category: Category) -> str:
        """Резервный ответ"""
        return (
            f"По запросу '{query}' (категория: {category}):\n\n"
            "Не удалось сформировать детальный ответ. "
            "Проверьте настройки API."
        )
    
    def _validate_safety(self, answer: str) -> str:
        """Проверка безопасности"""
        try:
            system_prompt = (
                "Проверь ответ на опасные рекомендации по 3D-печати. "
                "Если всё в порядке - верни БЕЗ ИЗМЕНЕНИЙ."
            )
            
            validated = pplx_chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Проверь:\n\n{answer}"}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return validated
            
        except Exception as e:
            print(f"⚠️ Ошибка валидации: {e}")
            return "⚠️ Соблюдайте технику безопасности.\n\n" + answer
    
    def query(
        self, 
        question: str, 
        top_k: int = 3,
        dialog_context: str = "",
        enable_validation: bool = True
    ) -> str:
        """Полная обработка запроса"""
        if not self.knowledge_base:
            return "❌ База знаний не загружена."
        
        try:
            category = self._classify_query(question)
            documents = self._search_documents(question, top_k)
            answer = self._generate_answer(question, category, documents, dialog_context)
            
            if enable_validation:
                answer = self._validate_safety(answer)
            
            return answer
            
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"


# Тестирование
if __name__ == "__main__":
    print("🧪 Тестирование RAG Pipeline\n")
    
    rag = RAGPipeline()
    
    if rag.knowledge_base:
        print(f"📚 База: {len(rag.knowledge_base)} документов\n")
        
        test_query = "проблемы с печатью ABS"
        print(f"❓ Запрос: {test_query}\n")
        
        response = rag.query(test_query, top_k=3, enable_validation=False)
        print("🤖 Ответ:")
        print(response)
    else:
        print("❌ База знаний пуста")
