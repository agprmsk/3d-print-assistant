import time
import json
import pathlib
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://3dtoday.ru"
OUT_DIR = pathlib.Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "3D-Assistant-Bot/1.0"}

def fetch_page(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.text

def parse_article(url: str) -> dict:
    html = fetch_page(url)
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
    content_blocks = soup.select("article, .article, .content")
    text = "\n".join(block.get_text(" ", strip=True) for block in content_blocks) or soup.get_text(" ", strip=True)
    return {
        "url": url,
        "title": title,
        "content": text,
        "tags": [],
        "category": "",
    }

def crawl_section(start_url: str, limit: int | None = None) -> list[dict]:
    articles: list[dict] = []
    # Здесь вы реализуете обход страниц раздела, вытаскивая ссылки на статьи
    # Ниже — псевдокод
    page_urls = [start_url]
    seen = set()
    while page_urls:
        url = page_urls.pop(0)
        if url in seen:
            continue
        seen.add(url)
        html = fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.select("a"):
            href = a.get("href", "")
            if "/articles/" in href or "/wiki/" in href:
                full = href if href.startswith("http") else BASE_URL + href
                try:
                    art = parse_article(full)
                    articles.append(art)
                    if limit and len(articles) >= limit:
                        return articles
                    time.sleep(1)
                except Exception:
                    continue
        # Здесь можно добавлять пагинацию
    return articles

def main():
    all_docs: list[dict] = []

    sections = [
        "https://3dtoday.ru/wiki/",
        "https://3dtoday.ru/articles/",
        # при необходимости добавьте ещё разделы
    ]
    for url in sections:
        docs = crawl_section(url, limit=None)  # можно временно ограничить, напр. 50
        all_docs.extend(docs)

    out_path = OUT_DIR / "3dtoday_raw.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
