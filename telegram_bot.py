import os
import logging
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler, ConversationHandler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from models import (
    get_session, Employee, Event, Task, TaskStatus, 
    Activity, activity_participants, EventType, ActivityType, 
    Session, Base, engine, GeneralInfo
)
from sqlalchemy import or_, and_
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

# Download all required NLTK data
required_nltk_data = ['punkt', 'stopwords', 'punkt_tab']
for item in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except LookupError:
        nltk.download(item, quiet=True)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# States for conversation handler
CHOOSING, TYPING_REPLY = range(2)

# Инициализация модели для классификации с улучшенной конфигурацией
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if torch.cuda.is_available() else -1,
        framework="pt",
        top_k=3,  # Получаем топ-3 предсказания для лучшего анализа
        batch_size=1,  # Оптимальный размер батча для точности
        max_length=512,  # Увеличенная длина контекста
        truncation=True
    )
except Exception as e:
    logger.error(f"Error initializing classifier: {e}")
    # Fallback to a simpler model if the main one fails
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,  # Force CPU
        framework="pt",
        top_k=3
    )

# Define categories for classification
categories = [
    "поиск сотрудника",
    "информация о мероприятии",
    "информация о задаче",
    "социальные активности",
    "приветствие",
    "общая информация",
    "неопределенный запрос"
]

# Расширенные гипотезы для классификации с акцентом на поиск по навыкам
classification_hypotheses = {
    "поиск сотрудника": [
        "Это запрос о поиске сотрудников с определенными навыками или знаниями",
        "Это запрос о том, кто может помочь с определенной задачей",
        "Это запрос о поиске экспертов в определенной области",
        "Это запрос о том, кто знает или умеет что-то конкретное",
        "Это запрос о поиске специалистов по определенному направлению",
        "Это запрос о том, кто может поделиться опытом или знаниями",
        "Это запрос о поиске наставника или консультанта"
    ],
    "информация о мероприятии": [
        "Это запрос о корпоративном мероприятии или событии",
        "Это запрос о расписании или планах компании",
        "Это запрос о предстоящих встречах или активностях",
        "Это запрос о датах или времени проведения мероприятий",
        "Это запрос о деталях корпоративных событий"
    ],
    "информация о задаче": [
        "Это запрос о рабочей задаче или проекте",
        "Это запрос о статусе выполнения работы",
        "Это запрос о дедлайнах или приоритетах задач",
        "Это запрос о распределении задач между сотрудниками",
        "Это запрос о прогрессе или результатах работы"
    ],
    "социальные активности": [
        "Это запрос о внерабочих активностях сотрудников",
        "Это запрос о хобби, интересах или увлечениях коллег",
        "Это запрос о совместных мероприятиях или активностях",
        "Это запрос о групповых занятиях или играх",
        "Это запрос о досуге или развлечениях в компании"
    ],
    "приветствие": [
        "Это приветствие или начало разговора",
        "Это вежливое обращение к боту",
        "Это запрос на установление контакта",
        "Это прощание или завершение разговора",
        "Это выражение благодарности или признательности"
    ],
    "общая информация": [
        "Это запрос о структуре или организации компании",
        "Это запрос о политиках, правилах или процедурах",
        "Это запрос о корпоративной культуре или ценностях",
        "Это запрос о ресурсах или возможностях компании",
        "Это запрос о справочной информации"
    ],
    "неопределенный запрос": [
        "Это запрос, требующий уточнения или дополнительной информации",
        "Это неясный или неоднозначный запрос",
        "Это запрос, не относящийся к известным категориям",
        "Это запрос, который сложно классифицировать",
        "Это запрос, который может относиться к нескольким категориям"
    ]
}

# Расширенные словари для улучшенной классификации
query_variations = {
    # Отделы и команды
    "it": ["it", "айти", "информационные технологии", "разработка", "программирование", "технический отдел", "тех отдел"],
    "отдел": ["отдел", "отделе", "департамент", "команда", "группа", "подразделение", "направление"],
    "hr": ["hr", "эйчар", "кадры", "персонал", "отдел кадров", "hr отдел", "эйчар отдел"],
    "маркетинг": ["маркетинг", "маркетинговый", "реклама", "продвижение", "pr", "пиар"],
    "продажи": ["продажи", "коммерческий", "sales", "сейлз", "отдел продаж"],
    
    # Действия и состояния
    "работает": ["работает", "работают", "трудится", "трудятся", "находится", "находятся", "состоит", "состоят"],
    "участники": ["участники", "участник", "члены", "член", "сотрудники", "сотрудник", "коллеги", "коллега"],
    "знает": ["знает", "умеет", "может", "разбирается", "понимает", "владеет", "имеет опыт", "опытен", "компетентен"],
    "помочь": ["помочь", "подсказать", "научить", "объяснить", "показать", "рассказать", "консультировать", "поддержать"],
    
    # Задачи и проекты
    "задачи": ["задачи", "задача", "дело", "дела", "проект", "проекты", "работа", "работы", "поручение", "поручения"],
    "мероприятия": ["мероприятия", "мероприятие", "события", "событие", "встречи", "встреча", "собрание", "собрания"],
    "активности": ["активности", "активность", "занятия", "занятие", "игры", "игра", "досуг", "развлечения"],
    
    # Навыки и компетенции
    "навыки": ["навыки", "умения", "знания", "опыт", "компетенции", "способности", "квалификация", "профессионализм"],
    "эксперт": ["эксперт", "специалист", "профессионал", "мастер", "гуру", "знаток", "консультант", "тренер"],
    
    # Временные периоды
    "сегодня": ["сегодня", "сейчас", "в данный момент", "на данный момент"],
    "завтра": ["завтра", "следующий день", "на следующий день"],
    "неделя": ["неделя", "неделю", "на неделе", "в течение недели", "на этой неделе"],
    "месяц": ["месяц", "месяц", "в течение месяца", "в этом месяце", "на месяц"],
    
    # Статусы и приоритеты
    "важно": ["важно", "срочно", "приоритетно", "критично", "необходимо", "нужно"],
    "готово": ["готово", "завершено", "выполнено", "сделано", "закончено", "окончено"],
    "в процессе": ["в процессе", "выполняется", "делается", "работает над", "занят", "в работе"],
    
    # Социальные активности
    "йога": ["йога", "йогой", "занятия йогой", "практика йоги"],
    "спорт": ["спорт", "спортом", "фитнес", "тренировки", "занятия спортом"],
    "игры": ["игры", "игра", "игровые", "гейминг", "игровые активности"],
    "обучение": ["обучение", "тренинг", "курс", "семинар", "вебинар", "мастер-класс"]
}

# Расширенные паттерны для точной классификации
category_patterns = {
    "поиск сотрудника": [
        # Поиск по отделу
        r"кто\s+(?:работает|трудится|находится)\s+в\s+(?:it|отделе|команде|департаменте)",
        r"сотрудники\s+(?:it|отдела|команды|департамента)",
        r"состав\s+(?:it|отдела|команды|департамента)",
        # Поиск по навыкам
        r"кто\s+(?:знает|умеет|может|разбирается)\s+в\s+(?:python|java|javascript|дизайн|маркетинг)",
        r"найти\s+(?:специалиста|эксперта)\s+по\s+(?:python|java|javascript|дизайн|маркетинг)",
        r"есть\s+ли\s+(?:кто-то|кто)\s+(?:кто|кто-то)\s+(?:знает|умеет|может)\s+(?:python|java|javascript|дизайн|маркетинг)",
        # Поиск по интересам
        r"кто\s+(?:занимается|увлекается|интересуется)\s+(?:йогой|спортом|играми|программированием)",
        r"найти\s+(?:коллег|сотрудников)\s+(?:для|по)\s+(?:йоге|спорту|играм|программированию)",
        # Общий поиск
        r"найти\s+(?:сотрудника|специалиста|эксперта)",
        r"поиск\s+(?:сотрудника|специалиста|эксперта)",
        r"кто\s+(?:может|готов)\s+помочь\s+с"
    ],
    "информация о мероприятии": [
        # Временные периоды
        r"какие\s+(?:мероприятия|события|встречи)\s+(?:будут|запланированы)\s+(?:сегодня|завтра|на неделе|в этом месяце)",
        r"когда\s+(?:будет|пройдет|состоится)\s+(?:мероприятие|событие|встреча)",
        # Типы мероприятий
        r"(?:расписание|календарь)\s+(?:мероприятий|событий|встреч)",
        r"что\s+(?:будет|запланировано)\s+(?:сегодня|завтра|на неделе)",
        r"информация\s+о\s+(?:мероприятии|событии|встрече)"
    ],
    "информация о задаче": [
        # Статусы и приоритеты
        r"какие\s+(?:задачи|проекты|работы)\s+(?:есть|назначены|в процессе)",
        r"статус\s+(?:задачи|проекта|работы)",
        r"когда\s+(?:нужно|следует)\s+(?:закончить|завершить|сдать)",
        # Дедлайны
        r"дедлайн\s+(?:для|по)\s+(?:задаче|проекту|работе)",
        r"сроки\s+(?:выполнения|сдачи)\s+(?:задачи|проекта|работы)",
        # Приоритеты
        r"приоритет\s+(?:задачи|проекта|работы)",
        r"важность\s+(?:задачи|проекта|работы)"
    ],
    "социальные активности": [
        # Типы активностей
        r"какие\s+(?:активности|занятия|игры)\s+(?:будут|запланированы)",
        r"кто\s+(?:занимается|увлекается|интересуется)\s+(?:йогой|спортом|играми)",
        # Участие
        r"присоединиться\s+к\s+(?:активности|занятию|игре)",
        r"участие\s+в\s+(?:активности|занятии|игре)",
        # Организация
        r"организовать\s+(?:активность|занятие|игру)",
        r"создать\s+(?:активность|занятие|игру)"
    ]
}

# Инициализация стеммера для русского языка
stemmer = SnowballStemmer("russian")

def analyze_query(query: str) -> dict:
    """Analyze query to extract entities and intent with improved accuracy."""
    try:
        logger.info(f"Analyzing query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Инициализация результата
        result = {
            'entities': {
                'skills': [],
                'departments': [],
                'event_types': [],
                'task_types': [],
                'activities': [],
                'priorities': [],
                'statuses': [],
                'dates': [],
                'topics': [],
                'categories': [],
                'assignees': [],
                'organizers': []
            },
            'intent': None,
            'confidence': 0.0
        }
        
        # Расширенные паттерны для распознавания временных выражений
        time_patterns = {
            'today': ['сегодня', 'сейчас', 'в данный момент', 'на данный момент'],
            'tomorrow': ['завтра', 'на следующий день', 'послезавтра'],
            'this_week': ['на этой неделе', 'в течение недели', 'до конца недели', 'на неделе'],
            'next_week': ['на следующей неделе', 'в следующую неделю'],
            'this_month': ['в этом месяце', 'до конца месяца', 'в течение месяца'],
            'next_month': ['в следующем месяце', 'в будущем месяце']
        }
        
        # Расширенные паттерны для распознавания намерений
        intent_patterns = {
            'search_employee': [
                'кто', 'найти', 'поиск', 'покажи', 'есть ли', 'может ли',
                'кто-нибудь', 'кто-то', 'кто из', 'кто в', 'кто на',
                'знает', 'умеет', 'может', 'разбирается', 'работает',
                'специалист', 'эксперт', 'разработчик', 'инженер'
            ],
            'search_event': [
                'когда', 'где', 'во сколько', 'время', 'дата',
                'мероприятие', 'встреча', 'событие', 'тренинг',
                'собрание', 'конференция', 'презентация',
                'будет', 'пройдет', 'состоится', 'запланировано'
            ],
            'search_task': [
                'задача', 'todo', 'что нужно', 'что сделать',
                'что делать', 'что осталось', 'что в работе',
                'статус', 'прогресс', 'дедлайн', 'срок',
                'приоритет', 'важность', 'срочность'
            ],
            'search_activity': [
                'активность', 'развлечение', 'чем заняться',
                'куда пойти', 'что делать', 'досуг',
                'игра', 'спорт', 'йога', 'тимбилдинг'
            ],
            'search_info': [
                'информация', 'как', 'где', 'что', 'когда',
                'почему', 'зачем', 'какой', 'какая', 'какие',
                'расскажи', 'подскажи', 'объясни'
            ]
        }
        
        # Анализ временных выражений
        for time_key, patterns in time_patterns.items():
            if any(pattern in query for pattern in patterns):
                result['entities']['dates'].append(time_key)
        
        # Анализ намерений
        for intent, patterns in intent_patterns.items():
            if any(pattern in query for pattern in patterns):
                result['intent'] = intent
                break
        
        # Анализ контекста
        context_words = {
            'skills': ['знает', 'умеет', 'может', 'разбирается', 'опыт', 'навыки'],
            'departments': ['отдел', 'департамент', 'команда', 'группа'],
            'event_types': ['мероприятие', 'встреча', 'событие', 'тренинг'],
            'task_types': ['задача', 'проект', 'работа', 'поручение'],
            'activities': ['активность', 'развлечение', 'досуг', 'игра']
        }
        
        for entity_type, words in context_words.items():
            if any(word in query for word in words):
                # Извлекаем существительные после контекстных слов
                words = query.split()
                for i, word in enumerate(words):
                    if word in context_words[entity_type] and i + 1 < len(words):
                        result['entities'][entity_type].append(words[i + 1])
        
        # Расчет уверенности
        confidence = 0.0
        total_entities = sum(len(entities) for entities in result['entities'].values())
        if total_entities > 0:
            confidence += 0.4
        if result['intent']:
            confidence += 0.3
        if len(query.split()) > 2:
            confidence += 0.3
        
        result['confidence'] = min(confidence, 1.0)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return {
            'entities': {
                'skills': [],
                'departments': [],
                'event_types': [],
                'task_types': [],
                'activities': [],
                'priorities': [],
                'statuses': [],
                'dates': [],
                'topics': [],
                'categories': [],
                'assignees': [],
                'organizers': []
            },
            'intent': None,
            'confidence': 0.0
        }

def preprocess_query(query: str) -> str:
    """Preprocess query with improved normalization and cleaning."""
    try:
        logger.info(f"Preprocessing query: {query}")
        
        # Нормализация регистра
        query = query.lower()
        
        # Удаление лишних пробелов
        query = ' '.join(query.split())
        
        # Удаление пунктуации
        query = ''.join(char for char in query if char.isalnum() or char.isspace())
        
        # Удаление стоп-слов
        stop_words = set(stopwords.words('russian'))
        query_tokens = query.split()
        query_tokens = [token for token in query_tokens if token not in stop_words]
        
        # Стемминг
        stemmer = SnowballStemmer('russian')
        query_tokens = [stemmer.stem(token) for token in query_tokens]
        
        # Сборка обратно в строку
        processed_query = ' '.join(query_tokens)
        
        logger.info(f"Processed query: {processed_query}")
        return processed_query
        
    except Exception as e:
        logger.error(f"Error preprocessing query: {e}")
        return query.lower().strip()

def classify_query(query: str) -> Tuple[str, float]:
    """
    Улучшенная классификация запроса с использованием нескольких методов анализа.
    """
    try:
        # Предобработка запроса
        processed_query = preprocess_query(query)
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Определяем базовую категорию на основе анализа
        if analysis['intent'] == 'search_event' or any(word in query.lower() for word in ['когда', 'где', 'во сколько', 'мероприятие', 'встреча', 'событие']):
            category = "информация о мероприятии"
            confidence = 0.9
        elif analysis['intent'] == 'search_task' or any(word in query.lower() for word in ['задача', 'todo', 'что нужно', 'что сделать']):
            category = "информация о задаче"
            confidence = 0.9
        elif analysis['intent'] == 'search_employee' or any(word in query.lower() for word in ['кто', 'найти', 'покажи', 'есть ли']):
            category = "поиск сотрудника"
            confidence = 0.9
        elif analysis['intent'] == 'search_activity' or any(word in query.lower() for word in ['активность', 'развлечение', 'чем заняться']):
            category = "социальные активности"
            confidence = 0.9
        elif analysis['intent'] == 'search_info' or any(word in query.lower() for word in ['информация', 'как', 'где', 'что', 'когда']):
            category = "общая информация"
            confidence = 0.9
        else:
            # Используем ML-классификатор как запасной вариант
            classification = classifier(
                processed_query,
                categories,
                hypothesis_template="Это запрос о {}"
            )
            
            # Получаем лучший результат
            best_label = classification['labels'][0]
            best_score = classification['scores'][0]
            
            # Повышаем уверенность на основе анализа
            confidence = min(1.0, best_score * analysis['confidence'])
            category = best_label
        
        # Проверяем уверенность
        if confidence < 0.4:  # Порог уверенности
            return "неопределенный запрос", 1.0
        
        return category, confidence
        
    except Exception as e:
        logger.error(f"Error in classify_query: {e}")
        return "неопределенный запрос", 1.0

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    keyboard = [
        [KeyboardButton("🔍 Поиск сотрудников"), KeyboardButton("📅 Мероприятия")],
        [KeyboardButton("📋 Задачи"), KeyboardButton("🎮 Активности")],
        [KeyboardButton("❓ Помощь"), KeyboardButton("👥 Моя команда")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    welcome_text = (
        "👋 Добро пожаловать в корпоративный бот!\n\n"
        "Я помогу вам:\n"
        "• Найти коллег по имени, отделу или навыкам\n"
        "• Узнать о предстоящих мероприятиях\n"
        "• Управлять задачами\n"
        "• Присоединиться к активностям\n\n"
        "Просто напишите ваш вопрос или используйте кнопки меню!\n\n"
        "Примеры вопросов:\n"
        "• Кто знает Python?\n"
        "• Какие мероприятия на этой неделе?\n"
        "• Покажи мои задачи\n"
        "• Какие активности сегодня?"
    )
    
    await update.message.reply_text(welcome_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *Как пользоваться ботом:*\n\n"
        "*Примеры вопросов о сотрудниках:*\n"
        "• 'Кто знает Python?'\n"
        "• 'Найди сотрудника из IT отдела'\n"
        "• 'Покажи всех разработчиков'\n"
        "• 'Когда день рождения у Марии Ивановой?'\n"
        "• 'Кто занимается йогой?'\n\n"
        "*Примеры вопросов о мероприятиях:*\n"
        "• 'Какие мероприятия на этой неделе?'\n"
        "• 'Когда тренинг по Python?'\n"
        "• 'Когда следующая встреча команды?'\n"
        "• 'Что будет завтра?'\n"
        "• 'Когда корпоративный обед?'\n"
        "*Примеры вопросов о задачах:*\n"
        "• 'Какие у меня задачи?'\n"
        "• 'Покажи мои задачи на этой неделе'\n"
        "• 'Какие задачи в работе?'\n"
        "• 'Какие задачи с высоким приоритетом?'\n"
        "• 'Когда дедлайн по проекту?'\n\n"
        "*Примеры вопросов об активностях:*\n"
        "• 'Какие активности сегодня?'\n"
        "• 'Кто играет в настольные игры?'\n"
        "• 'Когда следующее занятие йогой?'\n"
        "• 'Какие спортивные активности?'\n"
        "• 'Как присоединиться к тимбилдингу?'\n\n"
        "*Общие вопросы:*\n"
        "• 'Где находится офис?'\n"
        "• 'Как связаться с IT поддержкой?'\n"
        "• 'Какие правила в компании?'\n"
        "• 'Где найти базу знаний?'\n\n"
        "Используйте кнопки меню для быстрого доступа к функциям!"
    )
    await update.message.reply_text(help_text, parse_mode='Markdown')

def smart_fuzzy_search(items: list, query: str, fields: list) -> list:
    """Perform smart fuzzy search with improved accuracy and relevance."""
    try:
        logger.info(f"Performing fuzzy search for query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        query_tokens = query.split()
        
        # Инициализация результатов
        results = []
        scores = {}
        
        # Для каждого элемента
        for item in items:
            item_score = 0.0
            
            # Проверяем каждое поле
            for field in fields:
                if hasattr(item, field):
                    field_value = str(getattr(item, field)).lower()
                    field_tokens = field_value.split()
                    
                    # Проверяем точное совпадение
                    if query in field_value:
                        item_score += 1.0
                        continue
                    
                    # Проверяем частичные совпадения
                    for token in query_tokens:
                        if token in field_value:
                            item_score += 0.5
                        
                        # Проверяем похожесть токенов
                        for field_token in field_tokens:
                            similarity = calculate_similarity(token, field_token)
                            if similarity > 0.8:
                                item_score += similarity * 0.3
            
            # Если нашли совпадения
            if item_score > 0:
                scores[item] = item_score
        
        # Сортируем результаты по релевантности
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-5 результатов
        return [item for item, score in sorted_items[:5]]
        
    except Exception as e:
        logger.error(f"Error in fuzzy search: {e}")
        return []

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using Levenshtein distance."""
    try:
        # Если строки идентичны
        if str1 == str2:
            return 1.0
        
        # Если одна из строк пустая
        if not str1 or not str2:
            return 0.0
        
        # Вычисляем расстояние Левенштейна
        distance = levenshtein_distance(str1, str2)
        
        # Вычисляем максимальную длину
        max_len = max(len(str1), len(str2))
        
        # Вычисляем схожесть
        similarity = 1.0 - (distance / max_len)
        
        return similarity
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def levenshtein_distance(str1: str, str2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    try:
        # Создаем матрицу
        matrix = [[0 for _ in range(len(str2) + 1)] for _ in range(len(str1) + 1)]
        
        # Инициализируем первую строку и столбец
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j
        
        # Заполняем матрицу
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i - 1] == str2[j - 1]:
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(
                        matrix[i - 1][j] + 1,  # удаление
                        matrix[i][j - 1] + 1,  # вставка
                        matrix[i - 1][j - 1] + 1  # замена
                    )
        
        return matrix[len(str1)][len(str2)]
        
    except Exception as e:
        logger.error(f"Error calculating Levenshtein distance: {e}")
        return max(len(str1), len(str2))

def search_employees(query: str) -> str:
    """Search for employees based on query with improved accuracy."""
    try:
        logger.info(f"Searching employees with query: {query}")
        session = get_session()
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Базовый запрос
        base_query = session.query(Employee)
        
        # Извлекаем ключевые слова из запроса
        keywords = []
        if "django" in query:
            keywords.append("django")
        if "python" in query:
            keywords.append("python")
        if "flask" in query:
            keywords.append("flask")
        if "fastapi" in query:
            keywords.append("fastapi")
        
        # Применяем фильтры на основе анализа и ключевых слов
        if keywords:
            skill_conditions = []
            for keyword in keywords:
                skill_conditions.append(Employee.skills.ilike(f"%{keyword}%"))
            base_query = base_query.filter(or_(*skill_conditions))
        elif analysis['entities']['skills']:
            skill_conditions = []
            for skill in analysis['entities']['skills']:
                skill_conditions.append(Employee.skills.ilike(f"%{skill}%"))
            base_query = base_query.filter(or_(*skill_conditions))
        
        if analysis['entities']['departments']:
            dept_conditions = []
            for dept in analysis['entities']['departments']:
                dept_conditions.extend([
                    Employee.department.ilike(f"%{dept}%"),
                    Employee.position.ilike(f"%{dept}%")
                ])
            base_query = base_query.filter(or_(*dept_conditions))
        
        # Выполняем поиск
        employees = base_query.all()
        
        if not employees:
            # Если нет точных совпадений, используем нечеткий поиск
            all_employees = session.query(Employee).all()
            fuzzy_results = smart_fuzzy_search(
                all_employees,
                query,
                ["name", "surname", "position", "department", "skills", "interests"]
            )
            
            if fuzzy_results:
                response = "Точных совпадений не найдено. Возможно, вы искали:\n\n"
                for emp in fuzzy_results:
                    response += format_employee_info(emp)
                return response
            
            return "Сотрудники не найдены. Попробуйте изменить параметры поиска."
        
        # Форматируем результаты
        response = "Найденные сотрудники:\n\n"
        for emp in employees:
            response += format_employee_info(emp)
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching employees: {e}")
        return "Произошла ошибка при поиске сотрудников. Пожалуйста, попробуйте позже."
    finally:
        if 'session' in locals():
            session.close()

def format_employee_info(emp: Employee) -> str:
    """Format employee information in a user-friendly way."""
    interests_str = f"🎯 Интересы: {emp.interests}\n" if emp.interests else ""
    return (
        f"👤 *{emp.name} {emp.surname}*\n"
        f"💼 Должность: {emp.position}\n"
        f"🏢 Отдел: {emp.department}\n"
        f"🛠 Навыки: {emp.skills}\n"
        f"{interests_str}\n"
    )

def search_events(query: str, session) -> str:
    """Search for events with improved accuracy and relevance."""
    try:
        logger.info(f"Searching events with query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Базовый запрос
        base_query = session.query(Event)
        
        # Применяем фильтры на основе анализа
        if analysis['entities']['event_types']:
            event_conditions = []
            for event_type in analysis['entities']['event_types']:
                event_conditions.extend([
                    Event.title.ilike(f"%{event_type}%"),
                    Event.description.ilike(f"%{event_type}%")
                ])
            base_query = base_query.filter(or_(*event_conditions))
        
        # Поиск по ключевым словам
        if "тренинг" in query or "python" in query:
            base_query = base_query.filter(
                or_(
                    Event.title.ilike("%тренинг%"),
                    Event.title.ilike("%python%"),
                    Event.description.ilike("%тренинг%"),
                    Event.description.ilike("%python%")
                )
            )
        
        # Выполняем поиск
        events = base_query.all()
        
        if not events:
            # Если нет точных совпадений, используем нечеткий поиск
            all_events = session.query(Event).all()
            fuzzy_results = smart_fuzzy_search(
                all_events,
                query,
                ["title", "description", "location"]
            )
            
            if fuzzy_results:
                response = "Точных совпадений не найдено. Возможно, вы искали:\n\n"
                for event in fuzzy_results:
                    response += format_event_info(event)
                return response
            
            return "Мероприятия не найдены. Попробуйте изменить параметры поиска."
        
        # Форматируем результаты
        response = "Найденные мероприятия:\n\n"
        for event in events:
            response += format_event_info(event)
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching events: {e}")
        return "Произошла ошибка при поиске мероприятий. Пожалуйста, попробуйте позже."

def format_event_info(event: Event) -> str:
    """Format event information in a user-friendly way."""
    try:
        return (
            f"🎯 *{event.title}*\n"
            f"📅 Дата: {event.date}\n"
            f"⏰ Время: {event.time}\n"
            f"📍 Место: {event.location}\n"
            f"📝 {event.description}\n"
            f"🔹 Тип: {event.type.value}\n\n"
        )
    except Exception as e:
        logger.error(f"Error formatting event info: {e}")
        return f"Ошибка при форматировании информации о мероприятии: {str(e)}"

def search_tasks(session, query: str) -> str:
    """Search for tasks with improved accuracy and relevance."""
    try:
        logger.info(f"Searching tasks with query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Базовый запрос
        base_query = session.query(Task)
        
        # Применяем фильтры на основе анализа
        if analysis['entities']['priorities']:
            priority_conditions = []
            for priority in analysis['entities']['priorities']:
                priority_conditions.append(Task.priority.ilike(f"%{priority}%"))
            base_query = base_query.filter(or_(*priority_conditions))
        
        if analysis['entities']['statuses']:
            status_conditions = []
            for status in analysis['entities']['statuses']:
                status_conditions.append(Task.status.ilike(f"%{status}%"))
            base_query = base_query.filter(or_(*status_conditions))
        
        if analysis['entities']['dates']:
            date_conditions = []
            for date in analysis['entities']['dates']:
                date_conditions.append(Task.deadline.ilike(f"%{date}%"))
            base_query = base_query.filter(or_(*date_conditions))
        
        # Выполняем поиск
        tasks = base_query.all()
        
        if not tasks:
            # Если нет точных совпадений, используем нечеткий поиск
            all_tasks = session.query(Task).all()
            fuzzy_results = smart_fuzzy_search(
                all_tasks,
                query,
                ["title", "description", "priority", "status"]
            )
            
            if fuzzy_results:
                response = "Точных совпадений не найдено. Возможно, вы искали:\n\n"
                for task in fuzzy_results:
                    response += format_task_info(task)
                return response
            
            return "Задачи не найдены. Попробуйте изменить параметры поиска."
        
        # Форматируем результаты
        response = "Найденные задачи:\n\n"
        for task in tasks:
            response += format_task_info(task)
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching tasks: {e}")
        return "Произошла ошибка при поиске задач. Пожалуйста, попробуйте позже."

def format_task_info(task: Task) -> str:
    """Format task information in a user-friendly way."""
    return (
        f"📋 *{task.title}*\n"
        f"📝 {task.description}\n"
        f"📊 Приоритет: {task.priority}\n"
        f"📈 Статус: {task.status.value}\n"
        f"⏰ Срок: {task.deadline}\n"
        f"👤 Исполнитель: {task.assignee.name if task.assignee else 'Не назначен'}\n\n"
    )

def search_activities(session, query: str) -> str:
    """Search for activities with improved accuracy and relevance."""
    try:
        logger.info(f"Searching activities with query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Базовый запрос
        base_query = session.query(Activity)
        
        # Применяем фильтры на основе анализа
        if analysis['entities']['activities']:
            activity_conditions = []
            for activity in analysis['entities']['activities']:
                activity_conditions.extend([
                    Activity.title.ilike(f"%{activity}%"),
                    Activity.description.ilike(f"%{activity}%")
                ])
            base_query = base_query.filter(or_(*activity_conditions))
        
        if analysis['entities']['dates']:
            date_conditions = []
            for date in analysis['entities']['dates']:
                date_conditions.extend([
                    Activity.start_date.ilike(f"%{date}%"),
                    Activity.end_date.ilike(f"%{date}%")
                ])
            base_query = base_query.filter(or_(*date_conditions))
        
        # Выполняем поиск
        activities = base_query.all()
        
        if not activities:
            # Если нет точных совпадений, используем нечеткий поиск
            all_activities = session.query(Activity).all()
            fuzzy_results = smart_fuzzy_search(
                all_activities,
                query,
                ["title", "description", "location", "organizer"]
            )
            
            if fuzzy_results:
                response = "Точных совпадений не найдено. Возможно, вы искали:\n\n"
                for activity in fuzzy_results:
                    response += format_activity_info(activity)
                return response
            
            return "Активности не найдены. Попробуйте изменить параметры поиска."
        
        # Форматируем результаты
        response = "Найденные активности:\n\n"
        for activity in activities:
            response += format_activity_info(activity)
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching activities: {e}")
        return "Произошла ошибка при поиске активностей. Пожалуйста, попробуйте позже."

def format_activity_info(activity: Activity) -> str:
    """Format activity information in a user-friendly way."""
    end_date_str = f" - {activity.end_date}" if activity.end_date else ""
    return (
        f"🎯 *{activity.title}*\n"
        f"📅 Дата: {activity.start_date}{end_date_str}\n"
        f"📍 Место: {activity.location}\n"
        f"👥 Организатор: {activity.organizer}\n"
        f"📝 {activity.description}\n"
        f"👥 Участники: {activity.participants}\n\n"
    )

def search_general_info(session, query: str) -> str:
    """Search for general information with improved accuracy and relevance."""
    try:
        logger.info(f"Searching general info with query: {query}")
        
        # Нормализация запроса
        query = query.lower().strip()
        
        # Анализ запроса
        analysis = analyze_query(query)
        
        # Базовый запрос
        base_query = session.query(GeneralInfo)
        
        # Применяем фильтры на основе анализа
        if analysis['entities']['topics']:
            topic_conditions = []
            for topic in analysis['entities']['topics']:
                topic_conditions.extend([
                    GeneralInfo.title.ilike(f"%{topic}%"),
                    GeneralInfo.content.ilike(f"%{topic}%")
                ])
            base_query = base_query.filter(or_(*topic_conditions))
        
        if analysis['entities']['categories']:
            category_conditions = []
            for category in analysis['entities']['categories']:
                category_conditions.append(GeneralInfo.category.ilike(f"%{category}%"))
            base_query = base_query.filter(or_(*category_conditions))
        
        # Выполняем поиск
        info_items = base_query.all()
        
        if not info_items:
            # Если нет точных совпадений, используем нечеткий поиск
            all_info = session.query(GeneralInfo).all()
            fuzzy_results = smart_fuzzy_search(
                all_info,
                query,
                ["title", "content", "category"]
            )
            
            if fuzzy_results:
                response = "Точных совпадений не найдено. Возможно, вы искали:\n\n"
                for info in fuzzy_results:
                    response += format_general_info(info)
                return response
            
            return "Информация не найдена. Попробуйте изменить параметры поиска."
        
        # Форматируем результаты
        response = "Найденная информация:\n\n"
        for info in info_items:
            response += format_general_info(info)
        
        return response
        
    except Exception as e:
        logger.error(f"Error searching general info: {e}")
        return "Произошла ошибка при поиске информации. Пожалуйста, попробуйте позже."

def format_general_info(info: GeneralInfo) -> str:
    """Format general information in a user-friendly way."""
    return (
        f"📚 *{info.title}*\n"
        f"📝 {info.content}\n"
        f"🏷 Категория: {info.category}\n"
        f"📅 Обновлено: {info.updated_at}\n\n"
    )

def format_employee_results(results):
    if not results:
        return "😕 Не удалось найти сотрудников по вашему запросу."
    
    response = "👥 *Найденные сотрудники:*\n\n"
    for emp in results:
        response += (
            f"*{emp['name']}*\n"
            f"Должность: {emp['position']}\n"
            f"Отдел: {emp['department']}\n"
            f"Навыки: {emp['skills']}\n"
            f"Интересы: {emp['interests']}\n\n"
        )
    return response

def format_event_results(results):
    if not results:
        return "😕 Не удалось найти мероприятия по вашему запросу."
    
    response = "📅 *Предстоящие мероприятия:*\n\n"
    for event in results:
        response += (
            f"*{event['name']}*\n"
            f"Тип: {event['type']}\n"
            f"Дата: {event['date']}\n"
            f"Время: {event['time']}\n"
            f"Место: {event['location']}\n"
            f"{event['description']}\n\n"
        )
    return response

def format_task_results(results):
    if not results:
        return "😕 Не удалось найти задачи по вашему запросу."
    
    response = "📋 *Задачи:*\n\n"
    for task in results:
        response += (
            f"*{task['title']}*\n"
            f"Статус: {task['status']}\n"
            f"Приоритет: {task['priority']}\n"
            f"Дедлайн: {task['deadline'] or 'Не указан'}\n"
            f"{task['description']}\n\n"
        )
    return response

def format_activity_results(results):
    if not results:
        return "😕 Не удалось найти активности по вашему запросу."
    
    response = "🎮 *Доступные активности:*\n\n"
    for activity in results:
        response += (
            f"*{activity['name']}*\n"
            f"Тип: {activity['type']}\n"
            f"Дата: {activity['date']}\n"
            f"Время: {activity['time']}\n"
            f"Место: {activity['location']}\n"
            f"Макс. участников: {activity['max_participants']}\n"
            f"{activity['description']}\n\n"
        )
    return response

def format_general_results(results):
    response = ""
    if results['employees']:
        response += "👥 *Сотрудники:*\n" + format_employee_results(results['employees']) + "\n"
    if results['events']:
        response += "📅 *Мероприятия:*\n" + format_event_results(results['events']) + "\n"
    if results['tasks']:
        response += "📋 *Задачи:*\n" + format_task_results(results['tasks']) + "\n"
    if results['activities']:
        response += "🎮 *Активности:*\n" + format_activity_results(results['activities']) + "\n"
    
    return response if response else "😕 Не удалось найти информацию по вашему запросу."

def split_compound_query(query: str) -> List[str]:
    """Split compound query into separate parts."""
    # Нормализуем запрос
    query = query.lower().strip()
    
    # Определяем разделители
    separators = ["и", "а также", "также", "еще", "кроме того", "плюс"]
    
    # Ищем разделитель в запросе
    for separator in separators:
        if separator in query:
            # Разбиваем запрос на части
            parts = query.split(separator)
            # Очищаем части от лишних пробелов и возвращаем
            return [part.strip() for part in parts if part.strip()]
    
    # Если разделитель не найден, возвращаем исходный запрос
    return [query]

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages with improved flexibility and understanding."""
    query = update.message.text
    logger.info(f"Received message: {query}")
    
    # Нормализация запроса
    query_lower = query.lower().strip()
    
    # Получаем сессию базы данных
    session = get_session()
    try:
        # Классифицируем запрос
        category, confidence = classify_query(query)
        logger.info(f"Query classified as: {category} with confidence: {confidence}")
        
        # Обрабатываем запрос в зависимости от категории
        if category == "поиск сотрудника":
            response = search_employees(query)
        elif category == "информация о мероприятии":
            response = search_events(query, session)
        elif category == "информация о задаче":
            response = search_tasks(session, query)
        elif category == "социальные активности":
            response = search_activities(session, query)
        elif category == "общая информация":
            response = search_general_info(session, query)
        else:
            # Если категория не определена или уверенность низкая
            response = (
                "Извините, я не совсем понял ваш запрос. "
                "Попробуйте переформулировать или используйте кнопки меню для навигации.\n\n"
                "Примеры вопросов:\n"
                "• Кто знает Python?\n"
                "• Какие мероприятия на этой неделе?\n"
                "• Покажи мои задачи\n"
                "• Какие активности сегодня?"
            )
        
        # Отправляем ответ
        await update.message.reply_text(response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте еще раз или обратитесь к администратору."
        )
    finally:
        session.close()

def format_event_results(events, is_upcoming=False):
    """Format event results in a user-friendly way."""
    if not events:
        return "Мероприятия не найдены."
    
    response = "📅 *Предстоящие мероприятия:*\n\n" if not is_upcoming else "На указанную дату мероприятий не найдено. Вот ближайшие мероприятия:\n\n"
    
    for event in events:
        response += (
            f"*{event.title}*\n"
            f"📅 Дата: {event.start_date}"
            f"{f' - {event.end_date}' if event.end_date else ''}\n"
            f"📍 Место: {event.location}\n"
            f"👥 Организатор: {event.organizer}\n"
            f"📝 {event.description}\n"
            f"🔹 Тип: {event.type.value}\n\n"
        )
    
    return response

async def create_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle activity creation."""
    keyboard = [
        [InlineKeyboardButton("🎮 Игра", callback_data="activity_game"),
         InlineKeyboardButton("🍽 Обед", callback_data="activity_lunch")],
        [InlineKeyboardButton("🏃 Спорт", callback_data="activity_sport"),
         InlineKeyboardButton("🎯 Тимбилдинг", callback_data="activity_teambuilding")],
        [InlineKeyboardButton("❌ Отмена", callback_data="activity_cancel")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎮 *Создание активности*\n\n"
        "Выберите тип активности:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return CHOOSING

async def join_activity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle joining an activity."""
    query = update.callback_query
    await query.answer()
    
    activity_id = int(query.data.split('_')[1])
    session = get_session()
    try:
        activity = session.query(Activity).get(activity_id)
        if not activity:
            await query.edit_message_text("❌ Активность не найдена.")
            return
        
        if len(activity.participants) >= activity.max_participants:
            await query.edit_message_text("❌ К сожалению, все места заняты.")
            return
        
        # Add participant logic here
        await query.edit_message_text(
            f"✅ Вы успешно присоединились к активности '{activity.name}'!\n"
            f"Дата: {activity.date}\n"
            f"Время: {activity.time}\n"
            f"Место: {activity.location}"
        )
    finally:
        session.close()

def init_db():
    """Initialize database with test data."""
    try:
        logger.info("Starting database initialization...")
        
        # Create tables
        logger.info("Dropping existing tables...")
        Base.metadata.drop_all(engine)  # Drop existing tables
        
        logger.info("Creating fresh tables...")
        Base.metadata.create_all(engine)  # Create fresh tables
        
        # Create new session
        logger.info("Creating new session...")
        session = get_session()
        
        try:
            # Add test employees
            logger.info("Adding test employees...")
            employees = [
                Employee(
                    name="Иван",
                    surname="Петров",
                    department="IT",
                    position="Python Developer",
                    skills="Python, Django, Flask, SQL",
                    interests="Программирование, AI, машинное обучение"
                ),
                Employee(
                    name="Мария",
                    surname="Иванова",
                    department="IT",
                    position="Senior Python Developer",
                    skills="Python, FastAPI, PostgreSQL, Docker",
                    interests="Backend разработка, микросервисы"
                ),
                Employee(
                    name="Алексей",
                    surname="Смирнов",
                    department="Data Science",
                    position="Data Scientist",
                    skills="Python, Pandas, NumPy, Scikit-learn",
                    interests="Анализ данных, машинное обучение"
                ),
                Employee(
                    name="Елена",
                    surname="Козлова",
                    department="IT",
                    position="Python QA Engineer",
                    skills="Python, Pytest, Selenium, API Testing",
                    interests="Автоматизация тестирования, качество кода"
                )
            ]
            
            # Add employees one by one with logging
            for emp in employees:
                session.add(emp)
                logger.info(f"Added employee: {emp.name} {emp.surname}")
            
            # Commit employees first
            session.commit()
            logger.info("Committed employees to database")
            
            # Add test events
            logger.info("Adding test events...")
            today = datetime.now().date()
            events = [
                Event(
                    title="Тренинг по Python",
                    description="Продвинутые возможности Python для разработчиков",
                    date=today + timedelta(days=1),
                    time=datetime.strptime("14:00", "%H:%M").time(),
                    location="Конференц-зал 2",
                    type=EventType.TRAINING
                ),
                Event(
                    title="Ежедневная встреча команды",
                    description="Обсуждение текущих задач и планов",
                    date=today + timedelta(days=1),
                    time=datetime.strptime("10:00", "%H:%M").time(),
                    location="Конференц-зал 1",
                    type=EventType.MEETING
                ),
                Event(
                    title="Корпоративный обед",
                    description="Еженедельный обед для всей команды",
                    date=today + timedelta(days=1),
                    time=datetime.strptime("13:00", "%H:%M").time(),
                    location="Столовая",
                    type=EventType.HOLIDAY
                )
            ]
            
            # Add events one by one with logging
            for event in events:
                session.add(event)
                logger.info(f"Added event: {event.title} on {event.date}")
            
            # Commit events
            session.commit()
            logger.info("Committed events to database")
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            session.rollback()
            raise
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        if 'session' in locals():
            session.close()
            logger.info("Database session closed")

def main():
    """Start the bot."""
    try:
        # Initialize database
        logger.info("Starting bot initialization...")
        init_db()
        
        # Verify database initialization
        session = get_session()
        try:
            # Check total employees
            employee_count = session.query(Employee).count()
            logger.info(f"Database verification: Found {employee_count} employees")
            
            if employee_count == 0:
                logger.error("Database initialization failed: no employees found")
                raise Exception("Database initialization failed: no employees found")
            
            # List all employees for verification
            all_employees = session.query(Employee).all()
            logger.info("All employees in database:")
            for emp in all_employees:
                logger.info(f"Employee: {emp.name} {emp.surname}, Position: {emp.position}, Skills: {emp.skills}")
            
            # Verify Python employees specifically
            python_employees = session.query(Employee).filter(
                or_(
                    Employee.skills.ilike("%Python%"),
                    Employee.position.ilike("%Python%")
                )
            ).all()
            
            logger.info(f"Found {len(python_employees)} employees with Python skills")
            for emp in python_employees:
                logger.info(f"Python employee: {emp.name} {emp.surname}, Position: {emp.position}, Skills: {emp.skills}")
            
            if not python_employees:
                logger.error("Database initialization failed: no Python employees found")
                raise Exception("Database initialization failed: no Python employees found")
            
            # Check events
            event_count = session.query(Event).count()
            logger.info(f"Database verification: Found {event_count} events")
            
        finally:
            session.close()
        
        # Create the Application
        application = Application.builder().token('8181926764:AAE0RsZomH3bdhLnGqatSi5W7HH3fwjiEQQ').build()

        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        
        # Add message handler for text messages
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Add conversation handler for activity creation
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("create_activity", create_activity)],
            states={
                CHOOSING: [
                    CallbackQueryHandler(join_activity, pattern="^activity_")
                ],
                TYPING_REPLY: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
                ]
            },
            fallbacks=[CommandHandler("cancel", lambda u, c: ConversationHandler.END)]
        )
        application.add_handler(conv_handler)

        # Start the Bot
        logger.info("Starting bot polling...")
        application.run_polling()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == '__main__':
    main() 