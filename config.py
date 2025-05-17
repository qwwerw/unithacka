import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8181926764:AAE0RsZomH3bdhLnGqatSi5W7HH3fwjiEQQ')  # Using the token from the error message

# Database Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///corporate_bot.db')

# AI Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Application Settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
TIMEZONE = os.getenv('TIMEZONE', 'UTC')
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', 'ru')

# Security Settings
ADMIN_USER_IDS = [int(id.strip()) for id in os.getenv('ADMIN_USER_IDS', '').split(',') if id.strip()]

# Message Templates
WELCOME_MESSAGE = """
Добро пожаловать в корпоративного бота-ассистента! 🤖

Я помогу вам:
• Найти сотрудников по навыкам и отделам
• Узнать о предстоящих мероприятиях
• Управлять задачами
• Организовывать социальные активности
• Получать важные напоминания

Примеры вопросов:
• Кто знает Python?
• Какие мероприятия на этой неделе?
• Покажи мои задачи
• Какие активности сегодня?
• Когда день рождения у Марии?
• Кто свободен для встречи?

Используйте /help для получения дополнительной информации.
"""

HELP_MESSAGE = """
Доступные команды:
/start - Начать работу с ботом
/help - Показать это сообщение
/events - Показать предстоящие мероприятия
/tasks - Показать ваши задачи
/activities - Показать доступные активности
/birthdays - Показать дни рождения в этом месяце
/availability - Проверить занятость сотрудников

Вы также можете задавать вопросы в свободной форме:
• "Кто знает Python?"
• "Какие мероприятия на этой неделе?"
• "Покажи мои задачи"
• "Какие активности сегодня?"
• "Когда день рождения у Марии?"
• "Кто свободен для встречи?"
"""

# Error Messages
ERROR_MESSAGES = {
    'general': "Произошла ошибка. Пожалуйста, попробуйте позже.",
    'not_found': "К сожалению, я не нашел информацию по вашему запросу.",
    'invalid_query': "Извините, я не совсем понял ваш запрос. Можете переформулировать?",
    'permission_denied': "У вас нет прав для выполнения этого действия.",
    'database_error': "Произошла ошибка при работе с базой данных.",
    'model_error': "Произошла ошибка при обработке запроса.",
}

# Search Settings
SEARCH_SETTINGS = {
    'max_results': 5,
    'min_confidence': 0.5,
    'fuzzy_threshold': 0.8,
}

# Activity Settings
ACTIVITY_SETTINGS = {
    'max_participants': 20,
    'min_participants': 2,
    'reminder_hours': 24,
}

# Task Settings
TASK_SETTINGS = {
    'max_priority': 5,
    'reminder_hours': 48,
    'statuses': ['todo', 'in_progress', 'done', 'blocked'],
}

# Event Settings
EVENT_SETTINGS = {
    'max_participants': 50,
    'reminder_hours': 24,
    'types': ['meeting', 'training', 'team_building', 'presentation', 'other'],
} 