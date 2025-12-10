from typing import Literal
from .models import pplx_chat
from .embeddings_store import get_vector_store

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
    vectordb = get_vector_store()
    return vectordb.similarity_search(user_query, k=k)

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
    system_prompt = (
        "Ты проверяешь ответ по 3D-печати на безопасность. "
        "Найди явно опасные советы (слишком высокие температуры, взрывоопасные материалы, "
        "пренебрежение ТБ) и если они есть, перепиши ответ, убрав опасное и добавив предупреждения. "
        "Если всё в порядке, верни ответ как есть."
    )
    return pplx_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer},
        ],
        temperature=0.1,
        max_tokens=900,
    )
