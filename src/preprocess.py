"""Предобработка и нормализация собранных данных"""
import json
from pathlib import Path

try:
    from .config import RAW_DATA_DIR, PROCESSED_DATA_PATH
except ImportError:
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_PATH


def normalize():
    """Нормализация сырых данных в единый формат"""
    # Проверяем оба возможных формата файлов
    json_path = RAW_DATA_DIR / "3dtoday_articles.json"
    jsonl_path = RAW_DATA_DIR / "3dtoday_raw.jsonl"
    
    in_path = None
    if json_path.exists():
        in_path = json_path
        print(f"📄 Найден JSON файл: {json_path}")
    elif jsonl_path.exists():
        in_path = jsonl_path
        print(f"📄 Найден JSONL файл: {jsonl_path}")
    else:
        print(f"⚠️ Файлы данных не найдены:")
        print(f"   - {json_path}")
        print(f"   - {jsonl_path}")
        print("Сначала запустите: python -m src.scraper_3dtoday")
        return
    
    processed_count = 0
    
    try:
        # Открываем файл и определяем формат
        with open(in_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Пробуем распарсить как массив JSON
        try:
            data = json.loads(content)
            if isinstance(data, list):
                items = data
                print(f"✅ Загружен JSON массив ({len(items)} элементов)")
            else:
                items = [data]
        except json.JSONDecodeError:
            # Если не массив, пробуем как JSONL (построчно)
            items = []
            for line in content.split('\n'):
                if line.strip():
                    items.append(json.loads(line))
            print(f"✅ Загружен JSONL ({len(items)} элементов)")
        
        # Обработка и сохранение
        with open(PROCESSED_DATA_PATH, 'w', encoding='utf-8') as fout:
            for item in items:
                # Нормализация структуры
                doc = {
                    "id": item.get("url", f"doc_{processed_count}"),
                    "title": item.get("title", "Без заголовка"),
                    "content": item.get("content", ""),
                    "source_url": item.get("url", ""),
                    "category": item.get("category", ""),
                    "tags": item.get("tags", []),
                }
                
                # Пропускаем пустые документы
                if not doc["content"] or len(doc["content"]) < 100:
                    continue
                
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                processed_count += 1
        
        print(f"✅ Обработано {processed_count} документов")
        print(f"📁 Сохранено в: {PROCESSED_DATA_PATH}")
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🔄 Предобработка данных\n")
    normalize()
