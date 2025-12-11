# src/telegram_bot.py
import os
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле")

print("🚀 Запуск Telegram Bot...")
print("🔧 Инициализация RAG-системы...")

# Инициализация RAG системы
try:
    from src.rag_pipeline import RAGPipeline
    rag = RAGPipeline()
    print("✅ RAG-система готова!")
except ImportError:
    # Если класса нет, пробуем импортировать функцию
    from src.rag_pipeline import handle_user_query
    rag = None
    print("✅ RAG-функция импортирована!")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "👋 Привет! Я помощник по 3D-печати.\n\n"
        "Задай мне любой вопрос о 3D-печати, и я постараюсь помочь!\n\n"
        "Примеры вопросов:\n"
        "• Какую температуру использовать для PLA?\n"
        "• Как настроить скорость печати?\n"
        "• Почему появляются дефекты слоёв?"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    await update.message.reply_text(
        "ℹ️ Доступные команды:\n\n"
        "/start - Приветствие и информация\n"
        "/help - Справка\n\n"
        "Просто отправь мне свой вопрос о 3D-печати, "
        "и я найду релевантную информацию!"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    user_query = update.message.text
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"

    print(f"\n📩 Получен вопрос от @{username} (ID: {user_id})")
    print(f"❓ Вопрос: {user_query}")

    # Отправляем индикатор "печатает..."
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    try:
        # Получаем ответ от RAG-системы
        if rag is not None:
            # Используем класс RAGPipeline
            # Пробуем разные возможные имена методов
            if hasattr(rag, 'query'):
                result = rag.query(user_query)
            elif hasattr(rag, 'get_answer'):
                result = rag.get_answer(user_query)
            elif hasattr(rag, 'answer'):
                result = rag.answer(user_query)
            elif hasattr(rag, 'handle_query'):
                result = rag.handle_query(user_query)
            else:
                raise AttributeError(f"Класс RAGPipeline не имеет известного метода для запросов. Доступные методы: {[m for m in dir(rag) if not m.startswith('_')]}")
        else:
            # Используем функцию handle_user_query
            result = handle_user_query(user_query)
        
        # Обрабатываем результат
        if isinstance(result, dict):
            answer = result.get("answer", str(result))
            sources = result.get("sources", [])
        else:
            answer = str(result)
            sources = []
        
        # Формируем ответ с источниками
        response = answer
        if sources and len(sources) > 0:
            response += "\n\n📚 Источники:\n" + "\n".join(f"• {s}" for s in sources[:3])

        print(f"✅ Ответ отправлен ({len(response)} символов)")

        # Отправляем ответ пользователю
        await update.message.reply_text(response)

    except Exception as e:
        error_message = (
            "❌ Извините, произошла ошибка при обработке вашего запроса. "
            "Попробуйте переформулировать вопрос."
        )
        await update.message.reply_text(error_message)
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    print(f"❌ Произошла ошибка: {context.error}")


def main():
    """Точка входа - создаёт event loop и запускает бота"""
    print("\n✅ Telegram Bot запущен и готов к работе!")
    print("🤖 Найдите своего бота в Telegram и начните общение\n")

    # Создаём приложение
    application = Application.builder().token(BOT_TOKEN).build()

    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_error_handler(error_handler)

    # Для Python 3.14: создаём event loop явно перед запуском
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Запускаем polling
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
