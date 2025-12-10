import requests
from typing import List, Dict, Any
from .config import PERPLEXITY_API_KEY, PPLX_MODEL_GENERAL

PPLX_URL = "https://api.perplexity.ai/chat/completions"

def pplx_chat(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
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
    resp = requests.post(PPLX_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
