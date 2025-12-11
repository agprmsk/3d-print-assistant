# src/scraper_3dtoday.py
import requests
from bs4 import BeautifulSoup
import time
import json
import os
from typing import List, Dict

class WikiScraper3DToday:
    """Скрапер для сбора статей с 3DToday Wiki"""
    
    def __init__(self, base_url: str = "https://3dtoday.ru"):
        self.base_url = base_url
        self.wiki_url = f"{base_url}/wiki"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.articles = []
    
    def get_article_links(self, max_articles: int = 50) -> List[str]:
        """
        Получение ссылок на статьи из главной страницы Wiki
        
        Args:
            max_articles: Максимальное количество статей для сбора
        
        Returns:
            Список URL статей
        """
        print(f"🔍 Сбор ссылок на статьи из {self.wiki_url}...")
        
        try:
            response = requests.get(self.wiki_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Поиск всех ссылок на статьи
            article_links = []
            
            # Ищем ссылки в различных элементах страницы
            selectors = [
                'a[href*="/wiki/"]',  # Все ссылки содержащие /wiki/
                '.wiki-list a',
                '.article-list a',
                '.content a[href^="/wiki/"]'
            ]
            
            for selector in selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href', '')
                    if href and href.startswith('/wiki/') and href != '/wiki' and href != '/wiki/':
                        full_url = f"{self.base_url}{href}"
                        if full_url not in article_links:
                            article_links.append(full_url)
                    
                    if len(article_links) >= max_articles:
                        break
                
                if len(article_links) >= max_articles:
                    break
            
            print(f"✓ Найдено {len(article_links)} ссылок на статьи")
            return article_links[:max_articles]
            
        except Exception as e:
            print(f"❌ Ошибка при получении ссылок: {e}")
            return []
    
    def scrape_article(self, url: str) -> Dict:
        """
        Извлечение контента статьи
        
        Args:
            url: URL статьи
        
        Returns:
            Словарь с данными статьи
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Извлечение заголовка
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else "Без названия"
            
            # Извлечение основного контента
            content_selectors = [
                '.wiki-content',
                '.article-content',
                '.content',
                'article',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Удаляем скрипты и стили
                    for tag in content_elem(['script', 'style', 'nav', 'footer']):
                        tag.decompose()
                    
                    # Получаем текст
                    content = content_elem.get_text(separator='\n', strip=True)
                    if len(content) > 100:  # Минимальная длина контента
                        break
            
            # Определение категории из URL
            category = "Общее"
            url_parts = url.split('/')
            if len(url_parts) > 4:
                category = url_parts[4].replace('-', ' ').title()
            
            return {
                'title': title,
                'url': url,
                'category': category,
                'content': content,
                'content_length': len(content)
            }
            
        except Exception as e:
            print(f"   ❌ Ошибка при обработке {url}: {e}")
            return None
    
    def scrape_articles(self, max_articles: int = 50, delay: float = 2.0):
        """
        Сбор статей с задержкой между запросами
        
        Args:
            max_articles: Максимальное количество статей
            delay: Задержка между запросами (секунды)
        """
        # Получение ссылок
        article_urls = self.get_article_links(max_articles)
        
        if not article_urls:
            print("❌ Не удалось получить ссылки на статьи")
            return
        
        print(f"\n📥 Начинаем сбор {len(article_urls)} статей...")
        
        for i, url in enumerate(article_urls, 1):
            print(f"\n[{i}/{len(article_urls)}] Обработка: {url}")
            
            article = self.scrape_article(url)
            
            if article and article['content_length'] > 100:
                self.articles.append(article)
                print(f"   ✓ Собрано {article['content_length']} символов")
            else:
                print(f"   ⚠ Пропущено (недостаточно контента)")
            
            # Задержка между запросами
            if i < len(article_urls):
                time.sleep(delay)
        
        print(f"\n✅ Собрано {len(self.articles)} статей")
    
    def save_articles(self, output_file: str = "data/raw/3dtoday_articles.json"):
        """Сохранение статей в JSON"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Данные сохранены в {output_file}")
        print(f"📊 Статистика:")
        print(f"   - Всего статей: {len(self.articles)}")
        
        if self.articles:
            total_chars = sum(a['content_length'] for a in self.articles)
            avg_chars = total_chars // len(self.articles)
            print(f"   - Всего символов: {total_chars:,}")
            print(f"   - Средняя длина: {avg_chars:,} символов")
    
    def run(self, max_articles: int = 50):
        """Запуск полного цикла сбора данных"""
        print("🚀 Запуск скрапера 3DToday Wiki\n")
        
        self.scrape_articles(max_articles=max_articles)
        self.save_articles()

if __name__ == "__main__":
    scraper = WikiScraper3DToday()
    # Собираем 90 статей с задержкой 2 секунды между запросами
    scraper.run(max_articles=90)
