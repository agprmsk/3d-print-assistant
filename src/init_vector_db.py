# src/init_vector_db.py
import json
import os
from embeddings_store_faiss import EmbeddingsStoreFAISS

def load_processed_data():
    """Загрузка обработанных данных"""
    data_files = [
        "data/processed_data.json",
        "data/test_dataset.json",
        "data/raw/3dtoday_articles.json"
    ]
    
    for data_path in data_files:
        if os.path.exists(data_path):
            print(f"📥 Загрузка данных из {data_path}...")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, data_path
    
    print("❌ Не найдены файлы с данными:")
    for path in data_files:
        print(f"   - {path}")
    return None, None

def init_vector_db():
    """Инициализация векторной базы данных"""
    print("🔧 Инициализация векторной базы данных...")
    
    # Загрузка данных
    data, source_path = load_processed_data()
    
    if not data:
        return
    
    # Получение статей
    if isinstance(data, dict):
        articles = data.get('articles', [])
    elif isinstance(data, list):
        articles = data
    else:
        print(f"❌ Неизвестный формат данных: {type(data)}")
        return
    
    print(f"✓ Загружено {len(articles)} статей из {source_path}")
    
    # Инициализация хранилища
    print("\n📦 Создание FAISS индекса...")
    store = EmbeddingsStoreFAISS()
    
    # Подготовка документов
    for article in articles:
        content = article.get('content') or article.get('text') or article.get('body')
        if content:
            metadata = {
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'category': article.get('category', '')
            }
            
            # Добавляем напрямую в списки
            store.documents.append(content)
            store.metadatas.append(metadata)
    
    if not store.documents:
        print("❌ Нет документов с контентом")
        print(f"Пример элемента: {articles[0] if articles else 'Нет элементов'}")
        return
    
    print(f"➕ Добавлено {len(store.documents)} документов")
    
    # Создание эмбеддингов и индекса
    print("🔄 Создание эмбеддингов и индекса...")
    embeddings = store.model.encode(
        store.documents,
        show_progress_bar=True,
        batch_size=32
    )
    
    # Создание FAISS индекса
    import faiss
    import numpy as np
    
    embeddings_np = np.array(embeddings).astype('float32')
    dimension = embeddings_np.shape[1]
    
    store.index = faiss.IndexFlatL2(dimension)
    store.index.add(embeddings_np)
    
    # Сохранение
    print("💾 Сохранение индекса...")
    store.save()
    
    print(f"\n✅ Векторная база данных успешно создана!")
    print(f"📊 Статистика:")
    print(f"   - Всего документов: {len(store.documents)}")
    print(f"   - Размер индекса: {store.index.ntotal}")
    print(f"   - Размерность: {dimension}")

if __name__ == "__main__":
    init_vector_db()
