# src/llm_client.py
import os
import requests
from typing import Optional, List, Dict

class PerplexityClient:
    """Клиент для работы с Perplexity API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация клиента
        
        Args:
            api_key: API ключ Perplexity (если не указан, берется из PERPLEXITY_API_KEY)
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("API ключ не найден. Укажите PERPLEXITY_API_KEY в .env")
        
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(
        self,
        prompt: str,
        model: str = "sonar",
        max_tokens: int = 500,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Генерация ответа через Perplexity API
        
        Args:
            prompt: Текст промпта
            model: Модель для генерации (sonar, sonar-pro)
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации (0-1)
            stream: Потоковый режим
        
        Returns:
            Сгенерированный текст
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка API запроса: {str(e)}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "sonar",
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Чат-режим с историей сообщений
        
        Args:
            messages: Список сообщений в формате [{"role": "user", "content": "..."}]
            model: Модель для генерации
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
        
        Returns:
            Ответ модели
        """
        endpoint = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка API запроса: {str(e)}")

# Тестирование
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Проверка API ключа
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("❌ Ошибка: PERPLEXITY_API_KEY не найден в .env")
        exit(1)
    
    print("✓ API ключ найден")
    
    # Инициализация клиента
    client = PerplexityClient(api_key=api_key)
    print("✓ Клиент инициализирован")
    
    # Тестовый запрос
    test_prompt = "Объясни кратко, что такое 3D-печать"
    print(f"\n❓ Вопрос: {test_prompt}")
    print("⏳ Генерация ответа...\n")
    
    try:
        answer = client.generate(
            prompt=test_prompt,
            max_tokens=200
        )
        print(f"✓ Ответ получен:\n{answer}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
