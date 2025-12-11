"""
RAG Pipeline - оптимизированная версия с FAISS и Perplexity API
"""
import os
import sys
import json
import time
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
    Оптимизированная RAG-система с несколькими агентами:
    1. Классификатор - определяет категорию запроса (по ключевым словам)
    2. Поисковик - ищет релевантные документы (FAISS)
    3. Консультант - формирует детальный ответ (Perplexity)
    4. Валидатор - проверяет безопасность (простая проверка слов)
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
        """Быстрая классификация по ключевым словам (БЕЗ LLM)"""
        query_lower = user_query.lower()
        
        # Диагностика дефектов
        if any(word in query_lower for word in ["забилось", "забилась", "сопло", "экструдер", "дефект", "слои", "полосы", "трещины", "расслаивается", "не прилипает", "отклеивается"]):
            return "диагностика_дефектов"
        
        # Подбор материала
        elif any(word in query_lower for word in ["материал", "pla", "abs", "petg", "filament", "пластик", "филамент", "tpu", "nylon"]):
            return "подбор_материала"
        
        # Настройка принтера
        elif any(word in query_lower for word in ["настройка", "калибровка", "температура", "скорость", "ретракт", "настроить", "откалибровать"]):
            return "настройка_принтера"
        
        # Слайсер
        elif any(word in query_lower for word in ["слайсер", "cura", "prusaslicer", "slicer", "нарезка"]):
            return "слайсер"
        
        # Основы / выбор принтера
        elif any(word in query_lower for word in ["начинаю", "новичок", "первый", "основы", "выбрать", "принтер", "какой принтер"]):
            return "основы"
        
        # Остальное
        else:
            return "другое"
    
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
        """Генерация детального ответа через Perplexity"""
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
            "Проверьте настройки API или попробуйте переформулировать вопрос."
        )
    
    def _validate_safety(self, answer: str) -> str:
        """Простая проверка опасных ключевых слов (БЕЗ LLM)"""
        dangerous_keywords = [
            "токсичн", "ядовит", "взрывоопасн", "взрыв", 
            "горюч", "легковоспламеня", "пожар", "отравлен"
        ]
        
        answer_lower = answer.lower()
        has_danger = any(keyword in answer_lower for keyword in dangerous_keywords)
        
        if has_danger:
            return "⚠️ БЕЗОПАСНОСТЬ: Соблюдайте технику безопасности при работе с материалами.\n\n" + answer
        
        return answer
    
    def query(
        self, 
        question: str, 
        top_k: int = 3,
        dialog_context: str = "",
        enable_validation: bool = True
    ) -> str:
        """Полная обработка запроса с логированием времени"""
        if not self.knowledge_base:
            return "❌ База знаний не загружена."
        
        # Проверка на тему 3D-печати
        relevant_keywords = [
            "принтер", "печат", "3d", "pla", "abs", "petg", "сопло", 
            "экструдер", "слайсер", "филамент", "модель", "слой"
        ]
        if not any(kw in question.lower() for kw in relevant_keywords):
            return (
                "Я специализируюсь на вопросах о 3D-печати. "
                "Пожалуйста, задайте вопрос по этой теме (например, о выборе принтера, "
                "настройке печати, устранении дефектов)."
            )
        
        try:
            start = time.time()
            
            # Классификация (БЕЗ LLM - мгновенно)
            t1 = time.time()
            category = self._classify_query(question)
            print(f"⏱️ Классификация: {time.time() - t1:.2f}s (категория: {category})")
            
            # Поиск документов (FAISS - быстро)
            t2 = time.time()
            documents = self._search_documents(question, top_k)
            print(f"⏱️ Поиск: {time.time() - t2:.2f}s (найдено: {len(documents)} док.)")
            
            # Генерация ответа (Perplexity - основная задержка)
            t3 = time.time()
            answer = self._generate_answer(question, category, documents, dialog_context)
            print(f"⏱️ Генерация: {time.time() - t3:.2f}s")
            
            # Валидация безопасности (БЕЗ LLM - мгновенно)
            if enable_validation:
                t4 = time.time()
                answer = self._validate_safety(answer)
                print(f"⏱️ Валидация: {time.time() - t4:.2f}s")
            
            print(f"⏱️ ИТОГО: {time.time() - start:.2f}s")
            return answer
            
        except Exception as e:
            print(f"❌ Ошибка обработки запроса: {e}")
            return f"❌ Ошибка: {str(e)}"


# Тест
