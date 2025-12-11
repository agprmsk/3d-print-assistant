from typing import Literal
from .models import pplx_chat
from .embeddings_store_faiss import get_vector_store

Category = Literal[
    "основы",
    "подбор_материала",
    "настройка_принтера",
    "диагностика_дефектов",
    "слайсер",
    "другое",
]

def classify_query(user_query: str) -> Category:
    system_prompt = (
        "Ты классификатор запросов по 3D-печати. "
        "Верни ОДНО слово из списка: основы, подбор_материала, настройка_принтера, "
        "диагностика_дефектов, слайсер, другое."
    )
    content = pplx_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0,
        max_tokens=10,
    )
    text = content.lower()
    for cat in ["основы", "подбор_материала", "настройка_принтера", "диагностика_дефектов", "слайсер"]:
        if cat in text:
            return cat  # type: ignore[return-value]
    return "другое"  # type: ignore[return-value]

def retrieve_knowledge(user_query: str, k: int = 6):
    """Получение релевантных документов из FAISS"""
    try:
        # Импортируем класс
        from .embeddings_store_faiss import EmbeddingsStoreFAISS
        
        # Создаём экземпляр и загружаем индекс
        store = EmbeddingsStoreFAISS()
        if not store.load():
            print("⚠️ FAISS индекс не найден")
            return []
        
        # Выполняем поиск
        results = store.search(user_query, k=k)
        print(f"✅ Найдено {len(results)} релевантных документов")
        
        # Преобразуем результаты в формат, ожидаемый consultant_answer
        # Создаём объекты с атрибутами page_content и metadata
        class Document:
            def __init__(self, text, metadata):
                self.page_content = text
                self.metadata = metadata
        
        documents = [Document(r['text'], r['metadata']) for r in results]
        return documents
        
    except Exception as e:
        print(f"⚠️ Ошибка FAISS поиска: {e}")
        import traceback
        traceback.print_exc()
        return []

def consultant_answer(user_query: str, docs, dialog_context: str = "") -> str:
    context_text = "\n\n".join(
        f"Заголовок: {d.metadata.get('title')}\nИсточник: {d.metadata.get('source_url')}\nТекст: {d.page_content}"
        for d in docs
    )
    system_prompt = (
        "Ты эксперт по 3D-печати. Используй ТОЛЬКО факты из предоставленного контекста. "
        "Если информации недостаточно, честно скажи об этом и предложи общие рекомендации. "
        "Давай конкретные параметры печати, типичные причины дефектов и шаги диагностики."
    )
    user_prompt = (
        f"Контекст диалога: {dialog_context}\n\n"
        f"База знаний:\n{context_text}\n\n"
        f"Вопрос пользователя: {user_query}\n\n"
        "Сформируй структурированный ответ: причины, параметры, шаги, ссылки."
    )
    return pplx_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=900,
    )
def safety_check(answer: str) -> str:
    """Простая проверка опасных ключевых слов"""
    dangerous_keywords = [
        "токсичн", "ядовит", "взрывоопасн", "взрыв",
        "горюч", "легковоспламеня", "пожар", "отравлен"
    ]
    
    answer_lower = answer.lower()
    has_danger = any(keyword in answer_lower for keyword in dangerous_keywords)
    
    if has_danger:
        return "⚠️ БЕЗОПАСНОСТЬ: Соблюдайте технику безопасности при работе с материалами.\n\n" + answer
    
    return answer

