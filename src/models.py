"""Клиент для работы с Perplexity API"""
import os
import requests
from typing import List, Dict, Any

# Импорт конфигурации с обработкой ошибок
try:
    from .config import PERPLEXITY_API_KEY, PPLX_MODEL_GENERAL
except ImportError:
    from src.config import PERPLEXITY_API_KEY, PPLX_MODEL_GENERAL

PPLX_URL = "https://api.perplexity.ai/chat/completions"


def pplx_chat(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """
    Отправить запрос к Perplexity API
    
    Args:
        messages: Список сообщений [{"role": "user", "content": "..."}, ...]
        model: Название модели (по умолчанию из config)
        temperature: Температура генерации (0.0-1.0)
        max_tokens: Максимальное количество токенов в ответе
    
    Returns:
        Текст ответа от модели
    """
    if not PERPLEXITY_API_KEY:
        raise ValueError(
            "PERPLEXITY_API_KEY не установлен. "
            "Создайте файл .env и добавьте: PERPLEXITY_API_KEY=ваш_ключ"
        )
    
    model = model or PPLX_MODEL_GENERAL
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        resp = requests.post(PPLX_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
        
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ошибка HTTP при обращении к Perplexity API: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ошибка соединения с Perplexity API: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Неожиданный формат ответа от Perplexity API: {e}")


# Тестирование
if __name__ == "__main__":
    print("🧪 Тест Perplexity API")
    print()
