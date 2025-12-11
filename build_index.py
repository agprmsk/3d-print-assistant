import sys
sys.path.insert(0, '.')

# Пробуем разные варианты импорта
try:
    from src.embeddings_store_faiss import EmbeddingsStoreFAISS
except ImportError:
    try:
        import src.embeddings_store_faiss as embeddings_module
        EmbeddingsStoreFAISS = embeddings_module.EmbeddingsStoreFAISS
    except:
        # Прямой импорт
        import importlib.util
        spec = importlib.util.spec_from_file_location("embeddings_store_faiss", "src/embeddings_store_faiss.py")
        embeddings_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(embeddings_module)
        EmbeddingsStoreFAISS = embeddings_module.EmbeddingsStoreFAISS

if __name__ == "__main__":
    print("🚀 Создание FAISS индекса...")
    
    store = EmbeddingsStoreFAISS(db_path="data/faiss_index")
    
    # Проверка файла
    import os
    articles_path = "data/raw/3dtoday_articles.json"
    
    if not os.path.exists(articles_path):
        print(f"❌ Файл {articles_path} не найден!")
    else:
        print(f"✅ Файл найден: {articles_path}")
        
        # Строим индекс
        store.build_from_articles(articles_path)
        
        # Тест
        print("\n🔍 Тестовый поиск:")
        results = store.search("Как печатать PLA пластиком?", k=2)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['metadata']['title']}")
            print(f"   {r['text'][:200]}...")
        
        print("\n✅ Готово! Индекс сохранён в data/faiss_index/")
