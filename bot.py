import os
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from models import init_db, get_session, Employee, Event, Task, TaskStatus
from sqlalchemy import or_, and_
import re
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the AI models with better configuration
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else -1
)

# Define categories for classification with examples and synonyms
categories = [
    "поиск сотрудника",
    "информация о мероприятии",
    "информация о задаче",
    "приветствие",
    "неопределенный запрос"
]

# Define example queries and synonyms for each category with improved patterns
category_patterns = {
    "поиск сотрудника": {
        "keywords": [
            'отдел', 'отделе', 'it', 'hr', 'sales', 'marketing', 'проект', 'project',
            'разработка', 'разработчик', 'менеджер', 'директор', 'руководитель',
            'специалист', 'инженер', 'аналитик', 'дизайнер', 'тестировщик'
        ],
        "synonyms": [
            'найти', 'показать', 'кто', 'какие', 'список', 'сотрудники', 'работники',
            'коллеги', 'люди', 'команда', 'группа', 'отдел', 'подразделение',
            'искать', 'поиск', 'найти', 'показать', 'вывести', 'отобразить'
        ],
        "examples": [
            "кто работает в отделе",
            "найти сотрудника",
            "кто из отдела",
            "покажи сотрудников",
            "кто работает над проектом",
            "список сотрудников",
            "какие люди работают",
            "кто в команде",
            "покажи команду разработки",
            "кто отвечает за проект",
            "найти специалиста по",
            "кто руководит отделом"
        ]
    },
    "информация о мероприятии": {
        "keywords": [
            'мероприятие', 'мероприятия', 'корпоратив', 'тренинг', 'встреча',
            'неделе', 'недели', 'месяц', 'месяца', 'день', 'дня', 'дата',
            'время', 'расписание', 'план', 'календарь', 'событие', 'события'
        ],
        "synonyms": [
            'когда', 'расписание', 'план', 'календарь', 'дата', 'время',
            'запланировано', 'назначено', 'будет', 'пройдет', 'состоится',
            'организовано', 'подготовлено', 'устроено'
        ],
        "examples": [
            "какие мероприятия",
            "когда корпоратив",
            "расписание мероприятий",
            "какие встречи",
            "когда тренинг",
            "что запланировано",
            "какие события",
            "что будет на неделе",
            "какие встречи запланированы",
            "расписание на месяц",
            "когда следующее мероприятие",
            "что готовится в отделе"
        ]
    },
    "информация о задаче": {
        "keywords": [
            'задача', 'задачи', 'дедлайн', 'проект', 'работа', 'поручение',
            'обязанность', 'функция', 'роль', 'ответственность', 'контроль',
            'проверка', 'тестирование', 'разработка', 'внедрение'
        ],
        "synonyms": [
            'сделать', 'выполнить', 'срок', 'статус', 'прогресс', 'ход',
            'продвижение', 'этап', 'стадия', 'фаза', 'процесс', 'работа',
            'дело', 'поручение', 'обязанность'
        ],
        "examples": [
            "какие задачи",
            "что нужно сделать",
            "какие дедлайны",
            "статус задачи",
            "когда сдать",
            "что в работе",
            "текущие задачи",
            "мои поручения",
            "что на контроле",
            "какие проекты в работе",
            "статус разработки",
            "ход выполнения"
        ]
    },
    "приветствие": {
        "keywords": [
            'привет', 'здравствуй', 'добрый', 'начать', 'помощь', 'хеллоу',
            'хай', 'здорово', 'приветствую', 'доброе', 'добрый'
        ],
        "synonyms": [
            'здравствуйте', 'доброе утро', 'добрый день', 'добрый вечер',
            'хеллоу', 'хай', 'приветствую', 'здорово', 'добро пожаловать',
            'рад видеть', 'как дела', 'как жизнь'
        ],
        "examples": [
            "привет",
            "здравствуй",
            "добрый день",
            "начать",
            "помощь",
            "как пользоваться",
            "что умеешь",
            "как дела",
            "доброе утро",
            "добрый вечер",
            "рад тебя видеть",
            "как жизнь"
        ]
    }
}

def preprocess_query(query: str) -> str:
    """Preprocess the query for better classification."""
    # Convert to lowercase
    query = query.lower()
    
    # Remove punctuation but keep important symbols
    query = re.sub(r'[^\w\s\-]', ' ', query)
    
    # Remove extra spaces
    query = ' '.join(query.split())
    
    # Remove common stop words
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'не', 'ни', 'но', 'а', 'или', 'что', 'как'}
    words = query.split()
    query = ' '.join(word for word in words if word not in stop_words)
    
    return query

def calculate_category_score(query: str, category: str) -> float:
    """Calculate a score for how well the query matches a category."""
    score = 0.0
    patterns = category_patterns[category]
    
    # Check keywords with higher weight for exact matches
    for keyword in patterns["keywords"]:
        if keyword in query:
            score += 0.4
        elif any(word.startswith(keyword) or keyword.startswith(word) for word in query.split()):
            score += 0.2
    
    # Check synonyms
    for synonym in patterns["synonyms"]:
        if synonym in query:
            score += 0.3
        elif any(word.startswith(synonym) or synonym.startswith(word) for word in query.split()):
            score += 0.15
    
    # Check examples with highest weight
    for example in patterns["examples"]:
        if example in query:
            score += 0.6
        elif any(word in example for word in query.split()):
            score += 0.3
    
    return score

def classify_query(query: str) -> Tuple[str, float]:
    """Classify the user query into one of the predefined categories with confidence score."""
    query = preprocess_query(query)
    logger.info(f"Processing query: {query}")
    
    # Calculate scores for each category
    category_scores = {
        category: calculate_category_score(query, category)
        for category in categories if category != "неопределенный запрос"
    }
    
    # Get the category with the highest score
    max_score_category = max(category_scores.items(), key=lambda x: x[1])
    
    # If the highest score is too low, use the AI model
    if max_score_category[1] < 0.3:
        logger.info("Using AI model for classification")
        result = classifier(query, categories)
        max_score_index = result['scores'].index(max(result['scores']))
        category = result['labels'][max_score_index]
        confidence = result['scores'][max_score_index]
        logger.info(f"AI model classified as: {category} with confidence {confidence:.2f}")
    else:
        category = max_score_category[0]
        confidence = max_score_category[1]
        logger.info(f"Rule-based classification: {category} with confidence {confidence:.2f}")
    
    # If confidence is too low, return "неопределенный запрос"
    if confidence < 0.2:
        return "неопределенный запрос", confidence
    
    return category, confidence

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_message = (
        "👋 Добро пожаловать в корпоративный бот!\n\n"
        "Я могу помочь вам с:\n"
        "🔍 Поиском сотрудников\n"
        "📅 Информацией о мероприятиях\n"
        "✅ Информацией о задачах\n\n"
        "Просто задайте вопрос в свободной форме!"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 Как пользоваться ботом:\n\n"
        "1. Задайте вопрос в свободной форме, например:\n"
        "   - 'Кто работает в IT отделе?'\n"
        "   - 'Какие мероприятия запланированы на этой неделе?'\n"
        "   - 'Какие задачи у Ивана Петрова?'\n\n"
        "2. Я проанализирую ваш вопрос и предоставлю релевантную информацию.\n\n"
        "3. Вы также можете использовать команды:\n"
        "   /start - Начать работу с ботом\n"
        "   /help - Показать это сообщение"
    )
    await update.message.reply_text(help_text)

def search_employees(query: str) -> str:
    """Search for employees based on the query."""
    session = get_session()
    query_lower = query.lower()
    logger.info(f"Searching employees with query: {query_lower}")
    
    try:
        # Определяем ключевые слова для поиска (русские и английские)
        role_keywords = {
            'разработка': [
                'разработка', 'разработчик', 'программист', 'код', 'кодить',
                'developer', 'programmer', 'coder', 'software', 'engineer'
            ],
            'руководство': [
                'руководитель', 'директор', 'менеджер', 'глава', 'начальник',
                'manager', 'director', 'head', 'lead', 'chief', 'senior'
            ],
            'тестирование': [
                'тестирование', 'тестировщик', 'qa', 'контроль качества',
                'tester', 'qa engineer', 'quality', 'testing'
            ],
            'дизайн': [
                'дизайн', 'дизайнер', 'ui', 'ux', 'интерфейс',
                'designer', 'ui/ux', 'interface', 'frontend'
            ],
            'аналитика': [
                'аналитик', 'анализ', 'исследование', 'исследователь',
                'analyst', 'researcher', 'research', 'analysis'
            ]
        }
        
        # Определяем отделы (русские и английские названия)
        departments = {
            'it': ['it', 'айти', 'информационные технологии', 'разработка', 'development'],
            'hr': ['hr', 'эйчар', 'кадры', 'персонал', 'human resources'],
            'sales': ['sales', 'продажи', 'сейлз', 'коммерция'],
            'marketing': ['marketing', 'маркетинг', 'реклама', 'продвижение']
        }
        
        # Извлекаем поисковые термины
        search_terms = []
        search_roles = []
        search_departments = []
        
        # Проверяем роли
        for role, keywords in role_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                search_roles.append(role)
                logger.info(f"Found role: {role}")
        
        # Проверяем отделы
        for dept, keywords in departments.items():
            if any(keyword in query_lower for keyword in keywords):
                search_departments.append(dept)
                logger.info(f"Found department: {dept}")
        
        # Формируем запрос
        query_filters = []
        
        # Если найдены роли
        if search_roles:
            role_conditions = []
            for role in search_roles:
                # Добавляем поиск по русским и английским терминам
                role_keywords_list = role_keywords[role]
                role_conditions.append(or_(
                    *[Employee.position.ilike(f'%{keyword}%') for keyword in role_keywords_list]
                ))
            query_filters.append(or_(*role_conditions))
        
        # Если найдены отделы
        if search_departments:
            dept_conditions = []
            for dept in search_departments:
                # Добавляем поиск по русским и английским названиям
                dept_keywords_list = departments[dept]
                dept_conditions.append(or_(
                    *[Employee.department.ilike(f'%{keyword}%') for keyword in dept_keywords_list]
                ))
            query_filters.append(or_(*dept_conditions))
        
        # Если нет конкретных критериев, ищем по всему тексту
        if not query_filters:
            query_filters.append(or_(
                Employee.name.ilike(f'%{query}%'),
                Employee.position.ilike(f'%{query}%'),
                Employee.department.ilike(f'%{query}%')
            ))
        
        # Выполняем поиск
        employees = session.query(Employee).filter(and_(*query_filters)).all()
        
        if employees:
            # Группируем сотрудников по отделам
            dept_employees = {}
            for emp in employees:
                if emp.department not in dept_employees:
                    dept_employees[emp.department] = []
                dept_employees[emp.department].append(emp)
            
            # Формируем ответ
            response = "Найдены следующие сотрудники:\n\n"
            for dept, emps in dept_employees.items():
                response += f"📌 {dept}:\n"
                for emp in emps:
                    response += f"• {emp.name} - {emp.position}\n"
                response += "\n"
            return response
        
        return "Сотрудники не найдены. Попробуйте уточнить критерии поиска."
    finally:
        session.close()

def search_events(query: str) -> str:
    """Search for events based on the query."""
    session = get_session()
    query_lower = query.lower()
    logger.info(f"Searching events with query: {query_lower}")
    
    try:
        # Определяем временной период
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Проверяем, есть ли в запросе упоминание недели
        if 'неделе' in query_lower or 'недели' in query_lower:
            events = session.query(Event).filter(
                Event.date >= week_start,
                Event.date <= week_end
            ).all()
        else:
            # Поиск по названию или типу
            events = session.query(Event).filter(
                or_(
                    Event.name.ilike(f'%{query}%'),
                    Event.type.ilike(f'%{query}%')
                )
            ).all()
        
        if events:
            return "Найдены следующие мероприятия:\n" + "\n".join(
                f"• {event.name} ({event.date}) - {event.description}"
                for event in sorted(events, key=lambda x: x.date)
            )
        return "Мероприятия не найдены."
    finally:
        session.close()

def search_tasks(query: str) -> str:
    """Search for tasks based on the query."""
    session = get_session()
    query_lower = query.lower()
    logger.info(f"Searching tasks with query: {query_lower}")
    
    try:
        # Search by title or assignee
        tasks = session.query(Task).join(Employee).filter(
            or_(
                Task.title.ilike(f'%{query}%'),
                Employee.name.ilike(f'%{query}%')
            )
        ).all()
        
        if tasks:
            return "Найдены следующие задачи:\n" + "\n".join(
                f"• {task.title} (Срок: {task.deadline}, Статус: {task.status.value}, Исполнитель: {task.assignee.name})"
                for task in tasks
            )
        return "Задачи не найдены."
    finally:
        session.close()

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages and respond accordingly."""
    query = update.message.text
    logger.info(f"Received message: {query}")
    
    category, confidence = classify_query(query)
    logger.info(f"Classified as: {category} with confidence {confidence:.2f}")
    
    if category == "неопределенный запрос":
        response = (
            "Извините, я не совсем понял ваш вопрос. Попробуйте переформулировать или используйте /help для получения подсказок.\n\n"
            "Примеры вопросов:\n"
            "• Кто работает в IT отделе?\n"
            "• Какие мероприятия запланированы на этой неделе?\n"
            "• Какие задачи у Ивана Петрова?"
        )
    elif category == "поиск сотрудника":
        response = search_employees(query)
    elif category == "информация о мероприятии":
        response = search_events(query)
    elif category == "информация о задаче":
        response = search_tasks(query)
    else:
        response = "Извините, я не совсем понял ваш вопрос. Попробуйте переформулировать или используйте /help для получения подсказок."
    
    logger.info(f"Sending response: {response}")
    await update.message.reply_text(response)

def main():
    """Start the bot."""
    # Initialize database
    init_db()
    
    # Create the Application
    application = Application.builder().token("8181926764:AAE0RsZomH3bdhLnGqatSi5W7HH3fwjiEQQ").build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 
