import os
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, ConversationHandler
from sentence_transformers import SentenceTransformer
from models import (
    get_session, Employee, Event, Task, TaskStatus, 
    Activity, activity_participants, EventType, ActivityType, 
    Session, Base, engine, GeneralInfo
)
from models import init_db  # Explicitly import init_db
from sqlalchemy import or_, and_, extract
import re
from typing import List, Dict, Tuple, Optional, Union
import json
import requests
from dotenv import load_dotenv
import torch
import difflib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict
from functools import lru_cache
from Levenshtein import distance as levenshtein_distance
import traceback
from dateutil import parser
import pytz
from config import (
    TELEGRAM_TOKEN, DATABASE_URL, MODEL_NAME, DEBUG, TIMEZONE,
    DEFAULT_LANGUAGE, ADMIN_USER_IDS, WELCOME_MESSAGE, HELP_MESSAGE,
    ERROR_MESSAGES, SEARCH_SETTINGS, ACTIVITY_SETTINGS, TASK_SETTINGS,
    EVENT_SETTINGS
)

# Download all required NLTK data
required_nltk_data = ['punkt', 'stopwords', 'punkt_tab']
for item in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except LookupError:
        nltk.download(item, quiet=True)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG if DEBUG else logging.INFO
)
logger = logging.getLogger(__name__)

# States for conversation handler
CHOOSING, TYPING_REPLY = range(2)

# Инициализация модели для семантического поиска
try:
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Successfully loaded model: {MODEL_NAME}")
except Exception as e:
    logger.error(f"Error initializing sentence transformer: {e}")
    model = None

# Define categories for classification
categories = [
    "поиск сотрудника",
    "информация о мероприятии",
    "информация о задаче",
    "социальные активности",
    "приветствие",
    "общая информация",
    "день рождения",
    "календарь занятости",
    "напоминания",
    "неопределенный запрос"
]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    try:
        await update.message.reply_text(WELCOME_MESSAGE)
    except Exception as e:
        logger.error(f"Error in start command: {e}")
        await update.message.reply_text(ERROR_MESSAGES['general'])

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help с примерами вопросов"""
    help_text = """🤖 Я корпоративный бот, который может помочь вам найти информацию о компании и сотрудниках.

📋 Вот примеры вопросов, которые вы можете задать:

👥 О сотрудниках:
• Кто работает в IT отделе?
• Найди сотрудника по имени Иван
• Какие навыки у Марии?
• Кто специализируется на Python?

📅 О мероприятиях:
• Какие мероприятия запланированы на этой неделе?
• Когда следующая встреча команды?
• Где будет проходить тренинг?
• Кто организатор мероприятия?

✅ О задачах:
• Какие у меня активные задачи?
• Какие задачи назначены на Ивана?
• Какие задачи с высоким приоритетом?
• Какие задачи нужно выполнить до конца недели?

🎯 О социальных активностях:
• Какие активности запланированы?
• Когда турнир по настольному теннису?
• Кто участвует в активностях?
• Какие активности в спортзале?

🎂 О днях рождения:
• У кого день рождения в этом месяце?
• Когда день рождения у Марии?
• Кто родился в мае?

📊 О занятости:
• Кто свободен на этой неделе?
• Когда Иван занят?
• Какая занятость в IT отделе?

💡 Общая информация:
• Какие правила работы в компании?
• Где находится офис?
• Как связаться с HR?

🔍 Вы можете задавать вопросы в свободной форме, и я постараюсь найти нужную информацию!

❓ Если не уверены, что спросить, просто напишите /start для получения приветственного сообщения."""
    
    try:
        await update.message.reply_text(help_text)
    except Exception as e:
        logger.error(f"Error in help command: {e}")
        await update.message.reply_text("Произошла ошибка при отправке справки. Попробуйте позже.")

def classify_query(query: str) -> Tuple[str, float]:
    """Классификация запроса с использованием семантического поиска"""
    try:
        if model is None:
            logger.error("Model is not initialized")
            return "поиск сотрудника", 0.5  # Возвращаем базовую категорию
        
        # Получаем эмбеддинги запроса
        query_embedding = model.encode(query)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        
        # Получаем эмбеддинги категорий
        category_embeddings = model.encode(categories)
        logger.debug(f"Category embeddings shape: {category_embeddings.shape}")
        
        # Преобразуем в тензоры PyTorch
        query_tensor = torch.tensor(query_embedding).unsqueeze(0)  # [1, dim]
        category_tensor = torch.tensor(category_embeddings)  # [num_categories, dim]
        
        # Вычисляем схожесть
        similarities = torch.nn.functional.cosine_similarity(
            query_tensor,
            category_tensor,
            dim=1
        )
        logger.debug(f"Similarities shape: {similarities.shape}")
        logger.debug(f"Similarities: {similarities}")
        
        # Находим наиболее подходящую категорию
        max_similarity, max_idx = torch.max(similarities, dim=0)
        max_idx = max_idx.item()  # Convert tensor to int
        logger.debug(f"Max similarity: {max_similarity.item()}, Max index: {max_idx}")
        
        # Проверяем, что индекс находится в пределах списка категорий
        if 0 <= max_idx < len(categories):
            category = categories[max_idx]
            confidence = max_similarity.item()
            logger.info(f"Classified query '{query}' as '{category}' with confidence {confidence:.2f}")
            return category, confidence
        else:
            logger.warning(f"Invalid category index {max_idx} for query: {query}")
            return "поиск сотрудника", 0.5  # Возвращаем базовую категорию
        
    except Exception as e:
        logger.error(f"Error in classify_query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "поиск сотрудника", 0.5  # Возвращаем базовую категорию

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик входящих сообщений с улучшенной классификацией и обработкой запросов"""
    try:
        query = update.message.text.lower()
        logger.info(f"Received query: {query}")
        
        # Классифицируем запрос
        category, confidence = classify_query(query)
        logger.info(f"Query classified as: {category} with confidence: {confidence}")
        
        session = get_session()
        try:
            response = ""
            
            if category == "поиск сотрудника":
                logger.info("Searching for employees")
                response = search_employees(query)
            elif category == "информация о мероприятии":
                logger.info("Searching for events")
                response = search_events(query, session)
            elif category == "информация о задаче":
                logger.info("Searching for tasks")
                response = search_tasks(session, query)
            elif category == "социальные активности":
                logger.info("Searching for activities")
                response = search_activities(session, query)
            elif category == "день рождения":
                logger.info("Searching for birthdays")
                response = search_birthdays(query, session)
            elif category == "календарь занятости":
                logger.info("Searching for availability")
                response = search_availability(query, session)
            elif category == "приветствие":
                logger.info("Sending welcome message")
                response = WELCOME_MESSAGE
            elif category == "общая информация":
                logger.info("Searching for general info")
                response = search_general_info(session, query)
            else:
                # Если категория не определена, пробуем все поиски
                logger.info("Trying all search methods")
                responses = []
                
                emp_response = search_employees(query)
                if emp_response != ERROR_MESSAGES['not_found']:
                    responses.append(emp_response)
                
                event_response = search_events(query, session)
                if event_response != ERROR_MESSAGES['not_found']:
                    responses.append(event_response)
                
                task_response = search_tasks(session, query)
                if task_response != ERROR_MESSAGES['not_found']:
                    responses.append(task_response)
                
                activity_response = search_activities(session, query)
                if activity_response != ERROR_MESSAGES['not_found']:
                    responses.append(activity_response)
                
                if responses:
                    response = "\n\n".join(responses)
                else:
                    response = "Я нашел следующую информацию:\n\n" + search_general_info(session, query)
            
            if not response or response == ERROR_MESSAGES['not_found']:
                response = "Я могу помочь вам найти информацию о:\n" + \
                          "👥 Сотрудниках\n" + \
                          "📅 Мероприятиях\n" + \
                          "✅ Задачах\n" + \
                          "🎯 Активностях\n" + \
                          "🎂 Днях рождения\n" + \
                          "📊 Занятости\n\n" + \
                          "Задайте вопрос, и я постараюсь найти нужную информацию!"
            
            logger.info(f"Generated response: {response[:100]}...")  # Log first 100 chars of response
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            await update.message.reply_text("Я могу помочь вам найти информацию о сотрудниках, мероприятиях, задачах и многом другом. Попробуйте задать вопрос по-другому!")
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        await update.message.reply_text("Я могу помочь вам найти информацию о сотрудниках, мероприятиях, задачах и многом другом. Попробуйте задать вопрос по-другому!")

def search_employees(query: str) -> str:
    """Улучшенный поиск сотрудников с использованием семантического поиска"""
    try:
        session = get_session()
        query_embedding = model.encode(query) if model else None
        
        # Получаем всех сотрудников
        employees = session.query(Employee).filter(Employee.is_active == True).all()
        
        # Создаем эмбеддинги для каждого сотрудника
        employee_embeddings = []
        for emp in employees:
            emp_text = f"{emp.name} {emp.position} {emp.department} {emp.skills}"
            emp_embedding = model.encode(emp_text) if model else None
            employee_embeddings.append((emp, emp_embedding))
        
        # Находим наиболее похожих сотрудников
        results = []
        for emp, emp_embedding in employee_embeddings:
            if query_embedding is not None and emp_embedding is not None:
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_embedding),
                    torch.tensor(emp_embedding),
                    dim=0
                )
                results.append((emp, similarity.item()))
        
        # Сортируем результаты по схожести
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Форматируем результаты
        if not results:
            return ERROR_MESSAGES['not_found']
        
        response = "Вот что я нашел:\n\n"
        for emp, similarity in results[:SEARCH_SETTINGS['max_results']]:
            response += format_employee_info(emp)
            response += f"\nРелевантность: {similarity:.2f}\n\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_employees: {e}")
        return ERROR_MESSAGES['general']

def format_employee_info(emp: Employee) -> str:
    """Форматирование информации о сотруднике"""
    return f"""👤 {emp.name} {emp.surname}
📋 Должность: {emp.position}
🏢 Отдел: {emp.department}
📧 Email: {emp.email}
📱 Телефон: {emp.phone or 'Не указан'}
💡 Навыки: {emp.skills or 'Не указаны'}
🎯 Интересы: {emp.interests or 'Не указаны'}"""

def search_events(query: str, session) -> str:
    """Поиск мероприятий"""
    try:
        # Получаем текущую дату
        now = datetime.now(pytz.timezone(TIMEZONE))
        
        # Ищем предстоящие мероприятия
        events = session.query(Event).filter(
            Event.start_time >= now,
            Event.status == 'active'
        ).order_by(Event.start_time).all()
        
        if not events:
            return "На ближайшее время мероприятий не запланировано."
        
        response = "Предстоящие мероприятия:\n\n"
        for event in events:
            response += format_event_info(event)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_events: {e}")
        return ERROR_MESSAGES['general']

def format_event_info(event: Event) -> str:
    """Форматирование информации о мероприятии"""
    return f"""📅 {event.title}
📝 {event.description or 'Описание отсутствует'}
🕒 Время: {event.start_time.strftime('%d.%m.%Y %H:%M')} - {event.end_time.strftime('%H:%M')}
📍 Место: {event.location or 'Не указано'}
👥 Организатор: {event.organizer.name} {event.organizer.surname}
👥 Участников: {len(event.participants)}/{event.max_participants or '∞'}\n\n"""

def search_tasks(session, query: str) -> str:
    """Поиск задач"""
    try:
        # Получаем текущую дату
        now = datetime.now(pytz.timezone(TIMEZONE))
        
        # Ищем активные задачи
        tasks = session.query(Task).filter(
            Task.status != TaskStatus.DONE,
            Task.due_date >= now
        ).order_by(Task.priority.desc(), Task.due_date).all()
        
        if not tasks:
            return "У вас нет активных задач."
        
        response = "Ваши задачи:\n\n"
        for task in tasks:
            response += format_task_info(task)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_tasks: {e}")
        return ERROR_MESSAGES['general']

def format_task_info(task: Task) -> str:
    """Форматирование информации о задаче"""
    status_emoji = {
        TaskStatus.TODO: "📝",
        TaskStatus.IN_PROGRESS: "🔄",
        TaskStatus.DONE: "✅",
        TaskStatus.BLOCKED: "⛔"
    }
    
    return f"""{status_emoji.get(task.status, "📋")} {task.title}
📝 {task.description or 'Описание отсутствует'}
👤 Исполнитель: {task.assignee.name} {task.assignee.surname}
📅 Срок: {task.due_date.strftime('%d.%m.%Y') if task.due_date else 'Не указан'}
⭐ Приоритет: {'⭐' * task.priority}\n\n"""

def search_activities(session, query: str) -> str:
    """Поиск социальных активностей"""
    try:
        # Получаем текущую дату
        now = datetime.now(pytz.timezone(TIMEZONE))
        
        # Ищем активные мероприятия
        activities = session.query(Activity).filter(
            Activity.start_time >= now,
            Activity.status == 'active'
        ).order_by(Activity.start_time).all()
        
        if not activities:
            return "На ближайшее время активностей не запланировано."
        
        response = "Доступные активности:\n\n"
        for activity in activities:
            response += format_activity_info(activity)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_activities: {e}")
        return ERROR_MESSAGES['general']

def format_activity_info(activity: Activity) -> str:
    """Форматирование информации об активности"""
    return f"""🎯 {activity.title}
📝 {activity.description or 'Описание отсутствует'}
🕒 Время: {activity.start_time.strftime('%d.%m.%Y %H:%M')} - {activity.end_time.strftime('%H:%M')}
📍 Место: {activity.location or 'Не указано'}
👥 Организатор: {activity.organizer.name} {activity.organizer.surname}
👥 Участников: {activity.current_participants}/{activity.max_participants or '∞'}\n\n"""

def search_birthdays(query: str, session) -> str:
    """Поиск дней рождения"""
    try:
        # Получаем текущую дату
        now = datetime.now(pytz.timezone(TIMEZONE))
        
        # Ищем дни рождения в текущем месяце
        employees = session.query(Employee).filter(
            Employee.is_active == True,
            extract('month', Employee.birthday) == now.month
        ).order_by(extract('day', Employee.birthday)).all()
        
        if not employees:
            return "В этом месяце нет дней рождения."
        
        response = "Дни рождения в этом месяце:\n\n"
        for emp in employees:
            response += f"""🎂 {emp.name} {emp.surname}
📅 {emp.birthday.strftime('%d.%m.%Y')}
🏢 Отдел: {emp.department}\n\n"""
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_birthdays: {e}")
        return ERROR_MESSAGES['general']

def search_availability(query: str, session) -> str:
    """Поиск занятости сотрудников"""
    try:
        # Получаем текущую дату
        now = datetime.now(pytz.timezone(TIMEZONE))
        
        # Ищем сотрудников и их мероприятия
        employees = session.query(Employee).filter(Employee.is_active == True).all()
        
        response = "Занятость сотрудников:\n\n"
        for emp in employees:
            # Получаем мероприятия сотрудника
            events = session.query(Event).join(
                Event.participants
            ).filter(
                Employee.id == emp.id,
                Event.start_time >= now,
                Event.end_time <= now + timedelta(days=7)
            ).all()
            
            response += f"""👤 {emp.name} {emp.surname}
🏢 Отдел: {emp.department}\n"""
            
            if events:
                response += "📅 Занят на мероприятиях:\n"
                for event in events:
                    response += f"• {event.title} ({event.start_time.strftime('%d.%m.%Y %H:%M')})\n"
            else:
                response += "✅ Свободен на этой неделе\n"
            
            response += "\n"
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_availability: {e}")
        return ERROR_MESSAGES['general']

def search_general_info(session, query: str) -> str:
    """Поиск общей информации"""
    try:
        # Получаем эмбеддинги запроса
        query_embedding = model.encode(query) if model else None
        
        # Получаем всю информацию
        info_items = session.query(GeneralInfo).filter(
            GeneralInfo.is_active == True
        ).all()
        
        # Создаем эмбеддинги для каждого элемента
        info_embeddings = []
        for item in info_items:
            item_text = f"{item.title} {item.content} {item.category}"
            item_embedding = model.encode(item_text) if model else None
            info_embeddings.append((item, item_embedding))
        
        # Находим наиболее похожие элементы
        results = []
        for item, item_embedding in info_embeddings:
            if query_embedding is not None and item_embedding is not None:
                similarity = torch.nn.functional.cosine_similarity(
                    torch.tensor(query_embedding),
                    torch.tensor(item_embedding),
                    dim=0
                )
                results.append((item, similarity.item()))
        
        # Сортируем результаты по схожести
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Форматируем результаты
        if not results:
            return ERROR_MESSAGES['not_found']
        
        response = "Вот что я нашел:\n\n"
        for item, similarity in results[:SEARCH_SETTINGS['max_results']]:
            response += f"""📌 {item.title}
📝 {item.content}
🏷️ Категория: {item.category}
⭐ Релевантность: {similarity:.2f}\n\n"""
        
        return response
        
    except Exception as e:
        logger.error(f"Error in search_general_info: {e}")
        return ERROR_MESSAGES['general']

def init_test_data():
    """Инициализация тестовых данных в базе данных"""
    try:
        session = get_session()
        
        # Проверяем, есть ли уже данные
        if session.query(Employee).first() is not None:
            logger.info("Database already contains data, skipping initialization")
            return
        
        logger.info("Initializing test data...")
        
        # Создаем тестовых сотрудников
        employees = [
            Employee(
                name="Иван",
                surname="Иванов",
                position="Разработчик",
                department="IT",
                email="ivan@company.com",
                phone="+7-999-123-45-67",
                skills="Python, SQL, Docker",
                interests="Программирование, чтение",
                birthday=datetime(1990, 5, 15),
                is_active=True
            ),
            Employee(
                name="Мария",
                surname="Петрова",
                position="HR-менеджер",
                department="HR",
                email="maria@company.com",
                phone="+7-999-765-43-21",
                skills="Рекрутинг, обучение",
                interests="Психология, путешествия",
                birthday=datetime(1988, 8, 20),
                is_active=True
            )
        ]
        
        # Создаем тестовые мероприятия
        events = [
            Event(
                title="Встреча команды",
                description="Еженедельная встреча команды разработки",
                start_time=datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=1),
                end_time=datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=1, hours=1),
                location="Конференц-зал",
                event_type=EventType.MEETING,
                organizer=employees[0],
                max_participants=10,
                status='active'
            )
        ]
        
        # Создаем тестовые задачи
        tasks = [
            Task(
                title="Обновить документацию",
                description="Обновить документацию по API",
                status=TaskStatus.TODO,
                priority=2,
                assignee=employees[0],
                creator=employees[1],
                due_date=datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=7)
            )
        ]
        
        # Создаем тестовые активности
        activities = [
            Activity(
                title="Турнир по настольному теннису",
                description="Еженедельный турнир по настольному теннису",
                activity_type=ActivityType.SPORTS,
                start_time=datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=2),
                end_time=datetime.now(pytz.timezone(TIMEZONE)) + timedelta(days=2, hours=2),
                location="Спортзал",
                organizer=employees[1],
                max_participants=8,
                status='active'
            )
        ]
        
        # Создаем общую информацию
        general_info = [
            GeneralInfo(
                title="Правила работы",
                content="Основные правила работы в компании",
                category="Правила",
                is_active=True
            )
        ]
        
        # Добавляем все в базу данных
        session.add_all(employees)
        session.add_all(events)
        session.add_all(tasks)
        session.add_all(activities)
        session.add_all(general_info)
        
        # Сохраняем изменения
        session.commit()
        logger.info("Test data initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing test data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        session.rollback()
    finally:
        session.close()

def main():
    """Основная функция запуска бота"""
    try:
        # Инициализация базы данных
        init_db()
        
        # Инициализация тестовых данных
        init_test_data()
        
        # Создание приложения
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Добавление обработчиков
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Запуск бота
        application.run_polling()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == '__main__':
    main() 