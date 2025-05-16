import os
import logging
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from models import init_db, get_session, Employee, Event, Task, TaskStatus, Activity, activity_participants
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
    "социальные активности",
    "приветствие",
    "общая информация",
    "неопределенный запрос"
]

# Define example queries and synonyms for each category with improved patterns
category_patterns = {
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
    },
    "поиск сотрудника": {
        "keywords": [
            'отдел', 'отделе', 'it', 'hr', 'sales', 'marketing', 'проект', 'project',
            'разработка', 'разработчик', 'менеджер', 'директор', 'руководитель',
            'специалист', 'инженер', 'аналитик', 'дизайнер', 'тестировщик',
            'кто', 'найти', 'показать', 'список', 'сотрудники', 'коллеги',
            'работает', 'трудится', 'занимается', 'отвечает', 'знает',
            'умеет', 'может', 'способен', 'опыт', 'навыки', 'умения'
        ],
        "synonyms": [
            'найти', 'показать', 'кто', 'какие', 'список', 'сотрудники', 'работники',
            'коллеги', 'люди', 'команда', 'группа', 'отдел', 'подразделение',
            'искать', 'поиск', 'найти', 'показать', 'вывести', 'отобразить',
            'работает', 'трудится', 'занимается', 'отвечает', 'знает',
            'умеет', 'может', 'способен', 'опыт', 'навыки', 'умения',
            'специалист', 'эксперт', 'профессионал', 'мастер', 'гуру'
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
            "кто руководит отделом",
            "кто занимается разработкой",
            "покажи всех сотрудников отдела",
            "кто знает python",
            "кто умеет работать с базами данных",
            "найти эксперта по тестированию",
            "кто может помочь с проектом",
            "кто имеет опыт в маркетинге",
            "покажи специалистов по дизайну"
        ]
    },
    "информация о мероприятии": {
        "keywords": [
            'мероприятие', 'мероприятия', 'корпоратив', 'тренинг', 'встреча',
            'неделе', 'недели', 'месяц', 'месяца', 'день', 'дня', 'дата',
            'время', 'расписание', 'план', 'календарь', 'событие', 'события',
            'день рождения', 'дни рождения', 'праздник', 'праздники',
            'конференция', 'семинар', 'вебинар', 'презентация', 'доклад',
            'выступление', 'обучение', 'курс', 'лекция', 'мастер-класс'
        ],
        "synonyms": [
            'когда', 'расписание', 'план', 'календарь', 'дата', 'время',
            'запланировано', 'назначено', 'будет', 'пройдет', 'состоится',
            'организовано', 'подготовлено', 'устроено', 'праздновать',
            'отмечать', 'поздравлять', 'чествовать', 'проводить',
            'организовывать', 'планировать', 'готовить', 'устраивать'
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
            "что готовится в отделе",
            "когда день рождения",
            "какие праздники",
            "когда конференция",
            "расписание тренингов",
            "какие семинары на этой неделе",
            "когда мастер-класс",
            "что запланировано на месяц",
            "какие мероприятия в офисе"
        ]
    },
    "информация о задаче": {
        "keywords": [
            'задача', 'задачи', 'дедлайн', 'проект', 'работа', 'поручение',
            'обязанность', 'функция', 'роль', 'ответственность', 'контроль',
            'проверка', 'тестирование', 'разработка', 'внедрение',
            'срок', 'статус', 'прогресс', 'выполнение', 'todo', 'in progress', 'done',
            'блокер', 'проблема', 'ошибка', 'баг', 'фича', 'улучшение',
            'оптимизация', 'рефакторинг', 'документация', 'отчет'
        ],
        "synonyms": [
            'сделать', 'выполнить', 'срок', 'статус', 'прогресс', 'ход',
            'продвижение', 'этап', 'стадия', 'фаза', 'процесс', 'работа',
            'дело', 'поручение', 'обязанность', 'контролировать',
            'проверять', 'отслеживать', 'мониторить', 'в работе',
            'текущие', 'к выполнению', 'сделано', 'выполнено',
            'заблокировано', 'проблема', 'ошибка', 'исправить',
            'улучшить', 'оптимизировать', 'переписать', 'документировать'
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
            "ход выполнения",
            "что нужно сделать до",
            "какие задачи у",
            "покажи задачи к выполнению",
            "какие задачи в работе",
            "покажи выполненные задачи",
            "есть ли блокеры",
            "какие проблемы",
            "статус проекта"
        ]
    },
    "социальные активности": {
        "keywords": [
            'обед', 'игра', 'игры', 'встреча', 'встречи', 'общение',
            'команда', 'командный', 'вместе', 'совместно', 'активность',
            'активности', 'досуг', 'отдых', 'развлечение', 'развлечения',
            'йога', 'спорт', 'фитнес', 'танцы', 'музыка', 'кино',
            'театр', 'концерт', 'выставка', 'музей', 'парк', 'прогулка',
            'вечеринка', 'праздник', 'корпоратив', 'тимбилдинг'
        ],
        "synonyms": [
            'поиграть', 'пообедать', 'встретиться', 'познакомиться',
            'пообщаться', 'провести время', 'отдохнуть', 'развлечься',
            'командная игра', 'совместный обед', 'групповая активность',
            'заняться спортом', 'позаниматься йогой', 'потанцевать',
            'сходить в кино', 'посетить выставку', 'погулять в парке',
            'отпраздновать', 'провести тимбилдинг', 'организовать вечеринку'
        ],
        "examples": [
            "кто хочет поиграть",
            "кто идет на обед",
            "кто хочет встретиться",
            "найти партнера для игры",
            "кто свободен на обед",
            "кто хочет пообщаться",
            "найти компанию для",
            "кто хочет присоединиться",
            "кто готов поиграть",
            "кто хочет пообедать вместе",
            "кто занимается йогой",
            "кто хочет в кино",
            "кто идет на выставку",
            "кто хочет в парк",
            "кто готов к тимбилдингу",
            "кто хочет на вечеринку",
            "кто занимается спортом",
            "кто танцует",
            "кто любит музыку",
            "кто хочет в театр"
        ]
    },
    "общая информация": {
        "keywords": [
            'что', 'как', 'где', 'когда', 'почему', 'зачем',
            'информация', 'справка', 'помощь', 'подсказка',
            'правила', 'политика', 'процедуры', 'процессы',
            'структура', 'организация', 'компания', 'офис',
            'рабочее место', 'оборудование', 'ресурсы',
            'документы', 'файлы', 'база знаний', 'wiki'
        ],
        "synonyms": [
            'расскажи', 'объясни', 'покажи', 'найди', 'дай',
            'информацию', 'справку', 'помощь', 'подсказку',
            'правила', 'политику', 'процедуры', 'процессы',
            'структуру', 'организацию', 'компанию', 'офис',
            'рабочее место', 'оборудование', 'ресурсы',
            'документы', 'файлы', 'базу знаний', 'wiki'
        ],
        "examples": [
            "как работает",
            "где находится",
            "когда открыто",
            "что нужно знать",
            "какие правила",
            "как пользоваться",
            "где найти",
            "как получить доступ",
            "что делать если",
            "как решить проблему",
            "где посмотреть",
            "как узнать",
            "что нового",
            "какие изменения",
            "как обновить",
            "где документация",
            "как настроить",
            "что требуется",
            "как начать",
            "где справка"
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
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'не', 'ни', 'но', 'а', 'или', 'что', 'как', 'когда', 'где', 'почему', 'зачем'}
    words = query.split()
    query = ' '.join(word for word in words if word not in stop_words)
    
    return query

def calculate_category_score(query: str, category: str) -> float:
    """Calculate a score for how well the query matches a category."""
    score = 0.0
    patterns = category_patterns[category]
    
    # Проверяем наличие ключевых слов
    for keyword in patterns["keywords"]:
        if keyword in query:
            score += 0.4
        elif any(word.startswith(keyword) or keyword.startswith(word) for word in query.split()):
            score += 0.2
    
    # Проверяем синонимы
    for synonym in patterns["synonyms"]:
        if synonym in query:
            score += 0.3
        elif any(word.startswith(synonym) or synonym.startswith(word) for word in query.split()):
            score += 0.15
    
    # Проверяем примеры
    for example in patterns["examples"]:
        if example in query:
            score += 0.6
        elif any(word in example for word in query.split()):
            score += 0.3
    
    # Дополнительные проверки для поиска сотрудников
    if category == "поиск сотрудника":
        if any(word in query for word in ['знает', 'умеет', 'может', 'навыки', 'опыт']):
            score += 1.0
        if any(word in query for word in ['python', 'java', 'javascript', 'react', 'django']):
            score += 1.0
        if any(word in query for word in ['кто', 'найти', 'показать', 'список']):
            score += 0.5
    
    # Дополнительные проверки для мероприятий
    if category == "информация о мероприятии":
        if any(word in query for word in ['неделе', 'недели', 'сегодня', 'завтра']):
            score += 1.0
        if any(word in query for word in ['мероприятия', 'события', 'встречи']):
            score += 0.5
    
    # Дополнительные проверки для задач
    if category == "информация о задаче":
        if any(word in query for word in ['задача', 'задачи', 'задачу', 'задач']):
            score += 0.5
        if any(word in query for word in ['сделать', 'выполнить', 'сделано', 'выполнено']):
            score += 0.5
        if any(word in query for word in ['в работе', 'текущие', 'к выполнению']):
            score += 0.5
        if any(word in query for word in ['todo', 'in progress', 'done']):
            score += 0.5
    
    # Дополнительные проверки для социальных активностей
    if category == "социальные активности":
        if any(word in query for word in ['игра', 'игры', 'поиграть', 'настольные']):
            score += 0.5
        if any(word in query for word in ['обед', 'пообедать', 'вместе']):
            score += 0.5
        if any(word in query for word in ['активность', 'активности']):
            score += 0.5
        if any(word in query for word in ['йога', 'спорт', 'фитнес', 'танцы']):
            score += 0.5
        if any(word in query for word in ['кино', 'театр', 'концерт', 'выставка']):
            score += 0.5
    
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
        "✅ Информацией о задачах\n"
        "🎮 Социальными активностями\n\n"
        "Просто задайте вопрос в свободной форме!\n\n"
        "Примеры вопросов:\n"
        "• Кто работает в IT отделе?\n"
        "• Какие мероприятия на этой неделе?\n"
        "• Какие задачи у Ивана Петрова?\n"
        "• Кто хочет поиграть в настольные игры?"
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 Как пользоваться ботом:\n\n"
        "1. Задайте вопрос в свободной форме, например:\n"
        "   - 'Кто работает в IT отделе?'\n"
        "   - 'Какие мероприятия запланированы на этой неделе?'\n"
        "   - 'Какие задачи у Ивана Петрова?'\n"
        "   - 'Кто хочет поиграть в настольные игры?'\n\n"
        "2. Я проанализирую ваш вопрос и предоставлю релевантную информацию.\n\n"
        "3. Вы также можете использовать команды:\n"
        "   /start - Начать работу с ботом\n"
        "   /help - Показать это сообщение\n\n"
        "4. Социальные функции:\n"
        "   - Поиск коллег для совместных активностей\n"
        "   - Информация о днях рождения\n"
        "   - Организация встреч и мероприятий"
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
        search_skills = []
        search_interests = []
        
        # Проверяем навыки (расширенный список)
        tech_skills = {
            'python': ['python', 'питон'],
            'java': ['java', 'джава'],
            'javascript': ['javascript', 'js', 'джаваскрипт'],
            'react': ['react', 'реакт'],
            'django': ['django', 'джанго'],
            'docker': ['docker', 'докер'],
            'postgresql': ['postgresql', 'postgres', 'постгрес'],
            'mongodb': ['mongodb', 'монго'],
            'selenium': ['selenium', 'селениум'],
            'pytest': ['pytest', 'питест'],
            'postman': ['postman', 'постман'],
            'jira': ['jira', 'джира'],
            'agile': ['agile', 'аджайл'],
            'scrum': ['scrum', 'скрам'],
            'fastapi': ['fastapi', 'фастапи']
        }
        
        # Проверяем навыки
        for skill, keywords in tech_skills.items():
            if any(keyword in query_lower for keyword in keywords):
                search_skills.append(skill)
                logger.info(f"Found skill: {skill}")
        
        # Проверяем интересы
        if 'йога' in query_lower:
            search_interests.append('йога')
        if 'игра' in query_lower or 'игры' in query_lower:
            search_interests.append('настольные игры')
        if 'путешествия' in query_lower:
            search_interests.append('путешествия')
        if 'танцы' in query_lower:
            search_interests.append('танцы')
        if 'теннис' in query_lower:
            search_interests.append('теннис')
        
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
        
        # Если найдены навыки
        if search_skills:
            skill_conditions = []
            for skill in search_skills:
                skill_conditions.append(Employee.skills.ilike(f'%{skill}%'))
            query_filters.append(or_(*skill_conditions))
        
        # Если найдены интересы
        if search_interests:
            interest_conditions = []
            for interest in search_interests:
                interest_conditions.append(Employee.interests.ilike(f'%{interest}%'))
            query_filters.append(or_(*interest_conditions))
        
        # Если найдены роли
        if search_roles:
            role_conditions = []
            for role in search_roles:
                role_keywords_list = role_keywords[role]
                role_conditions.append(or_(
                    *[Employee.position.ilike(f'%{keyword}%') for keyword in role_keywords_list]
                ))
            query_filters.append(or_(*role_conditions))
        
        # Если найдены отделы
        if search_departments:
            dept_conditions = []
            for dept in search_departments:
                dept_keywords_list = departments[dept]
                dept_conditions.append(or_(
                    *[Employee.department.ilike(f'%{keyword}%') for keyword in dept_keywords_list]
                ))
            query_filters.append(or_(*dept_conditions))
        
        # Если запрос содержит "все" или "всех", показываем всех сотрудников
        if 'все' in query_lower or 'всех' in query_lower:
            employees = session.query(Employee).all()
        # Если нет конкретных критериев, ищем по всему тексту
        elif not query_filters:
            employees = session.query(Employee).filter(or_(
                Employee.name.ilike(f'%{query}%'),
                Employee.position.ilike(f'%{query}%'),
                Employee.department.ilike(f'%{query}%'),
                Employee.interests.ilike(f'%{query}%'),
                Employee.skills.ilike(f'%{query}%')
            )).all()
        else:
            # Выполняем поиск с фильтрами
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
                    if emp.skills:
                        response += f"  🛠️ Навыки: {emp.skills}\n"
                    if emp.interests:
                        response += f"  🎯 Интересы: {emp.interests}\n"
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
        from datetime import datetime, timedelta
        
        # Определяем временной период
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Проверяем, есть ли в запросе упоминание сотрудника
        employee_name = None
        for word in query_lower.split():
            if len(word) > 3:  # Игнорируем короткие слова
                employee = session.query(Employee).filter(
                    Employee.name.ilike(f'%{word}%')
                ).first()
                if employee:
                    employee_name = employee.name
                    break
        
        # Формируем запрос
        if employee_name:
            # Если найден сотрудник, ищем мероприятия, связанные с ним
            events = session.query(Event).join(
                event_participants
            ).join(
                Employee
            ).filter(
                Employee.name == employee_name
            ).all()
        elif 'неделе' in query_lower or 'недели' in query_lower:
            # Если запрос о неделе, показываем мероприятия на текущую неделю
            events = session.query(Event).filter(
                Event.date >= week_start,
                Event.date <= week_end
            ).all()
        elif 'семинар' in query_lower or 'тренинг' in query_lower:
            # Если запрос о семинарах или тренингах
            events = session.query(Event).filter(
                Event.type == EventType.TRAINING
            ).all()
        else:
            # Поиск по названию или типу
            events = session.query(Event).filter(
                or_(
                    Event.name.ilike(f'%{query}%'),
                    Event.type.ilike(f'%{query}%'),
                    Event.description.ilike(f'%{query}%')
                )
            ).all()
        
        if events:
            # Группируем мероприятия по датам
            date_events = {}
            for event in events:
                if event.date not in date_events:
                    date_events[event.date] = []
                date_events[event.date].append(event)
            
            # Формируем ответ
            response = "Найдены следующие мероприятия:\n\n"
            for date, evts in sorted(date_events.items()):
                response += f"📅 {date}:\n"
                for event in evts:
                    response += f"• {event.name} ({event.type.value})\n"
                    if event.time:
                        response += f"  🕒 {event.time}\n"
                    if event.description:
                        response += f"  {event.description}\n"
                    if event.location:
                        response += f"  📍 {event.location}\n"
                    if event.participants:
                        response += f"  👥 Участники: {', '.join(p.name for p in event.participants)}\n"
                    response += "\n"
            return response
        
        return "Мероприятия не найдены."
    finally:
        session.close()

def search_tasks(query: str) -> str:
    """Search for tasks based on the query."""
    session = get_session()
    query_lower = query.lower()
    logger.info(f"Searching tasks with query: {query_lower}")
    
    try:
        # Проверяем, есть ли в запросе упоминание сотрудника
        employee_name = None
        for word in query_lower.split():
            if len(word) > 3:  # Игнорируем короткие слова
                employee = session.query(Employee).filter(
                    Employee.name.ilike(f'%{word}%')
                ).first()
                if employee:
                    employee_name = employee.name
                    break
        
        # Формируем запрос
        if employee_name:
            # Если найден сотрудник, ищем его задачи
            tasks = session.query(Task).join(Employee).filter(
                Employee.name == employee_name
            ).all()
        elif 'в работе' in query_lower or 'текущие' in query_lower:
            # Если запрос о задачах в работе
            tasks = session.query(Task).filter(
                Task.status == TaskStatus.IN_PROGRESS
            ).all()
        elif 'сделать' in query_lower or 'todo' in query_lower:
            # Если запрос о задачах к выполнению
            tasks = session.query(Task).filter(
                Task.status == TaskStatus.TODO
            ).all()
        elif 'сделано' in query_lower or 'выполнено' in query_lower or 'done' in query_lower:
            # Если запрос о выполненных задачах
            tasks = session.query(Task).filter(
                Task.status == TaskStatus.DONE
            ).all()
        elif 'блокер' in query_lower or 'блокеры' in query_lower or 'проблема' in query_lower:
            # Если запрос о блокерах
            tasks = session.query(Task).filter(
                Task.status == TaskStatus.BLOCKED
            ).all()
        else:
            # Поиск по названию, тегам или статусу
            tasks = session.query(Task).filter(
                or_(
                    Task.title.ilike(f'%{query}%'),
                    Task.tags.ilike(f'%{query}%'),
                    Task.status.ilike(f'%{query}%')
                )
            ).all()
        
        if tasks:
            # Группируем задачи по статусу
            status_tasks = {}
            for task in tasks:
                if task.status not in status_tasks:
                    status_tasks[task.status] = []
                status_tasks[task.status].append(task)
            
            # Формируем ответ
            response = "Найдены следующие задачи:\n\n"
            for status, tsk in status_tasks.items():
                response += f"📌 {status.value}:\n"
                for task in tsk:
                    response += f"• {task.title}\n"
                    if task.description:
                        response += f"  {task.description}\n"
                    response += f"  📅 Срок: {task.deadline}\n"
                    response += f"  👤 Исполнитель: {task.assignee.name}\n"
                    if task.tags:
                        response += f"  🏷️ Теги: {task.tags}\n"
                    response += "\n"
            return response
        
        return "Задачи не найдены."
    finally:
        session.close()

def search_activities(query: str) -> str:
    """Search for social activities based on the query."""
    session = get_session()
    query_lower = query.lower()
    logger.info(f"Searching activities with query: {query_lower}")
    
    try:
        from datetime import datetime, timedelta
        
        # Определяем временной период
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Проверяем, есть ли в запросе упоминание сотрудника
        employee_name = None
        for word in query_lower.split():
            if len(word) > 3:  # Игнорируем короткие слова
                employee = session.query(Employee).filter(
                    Employee.name.ilike(f'%{word}%')
                ).first()
                if employee:
                    employee_name = employee.name
                    break
        
        # Формируем запрос
        if employee_name:
            # Если найден сотрудник, ищем активности, связанные с ним
            activities = session.query(Activity).join(
                activity_participants
            ).join(
                Employee
            ).filter(
                Employee.name == employee_name,
                Activity.is_active == True
            ).all()
        elif 'все' in query_lower or 'всех' in query_lower:
            # Показываем все активные активности
            activities = session.query(Activity).filter(
                Activity.is_active == True
            ).all()
        elif 'неделе' in query_lower or 'недели' in query_lower:
            # Если запрос о неделе, показываем активности на текущую неделю
            activities = session.query(Activity).filter(
                Activity.date >= week_start,
                Activity.date <= week_end,
                Activity.is_active == True
            ).all()
        elif 'йога' in query_lower:
            # Если запрос о йоге
            activities = session.query(Activity).filter(
                Activity.type == ActivityType.TRAINING,
                Activity.name.ilike('%йога%'),
                Activity.is_active == True
            ).all()
        elif 'игра' in query_lower or 'игры' in query_lower:
            # Если запрос об играх
            activities = session.query(Activity).filter(
                Activity.type == ActivityType.GAME,
                Activity.is_active == True
            ).all()
        else:
            # Поиск по названию, типу или описанию
            activities = session.query(Activity).filter(
                and_(
                    Activity.is_active == True,
                    or_(
                        Activity.name.ilike(f'%{query}%'),
                        Activity.description.ilike(f'%{query}%'),
                        Activity.type.ilike(f'%{query}%'),
                        Activity.tags.ilike(f'%{query}%')
                    )
                )
            ).all()
        
        if activities:
            # Группируем активности по датам
            date_activities = {}
            for activity in activities:
                if activity.date not in date_activities:
                    date_activities[activity.date] = []
                date_activities[activity.date].append(activity)
            
            # Формируем ответ
            response = "Найдены следующие активности:\n\n"
            for date, acts in sorted(date_activities.items()):
                response += f"📅 {date}:\n"
                for activity in acts:
                    response += f"• {activity.name} ({activity.type.value})\n"
                    if activity.time:
                        response += f"  🕒 {activity.time}\n"
                    if activity.description:
                        response += f"  {activity.description}\n"
                    if activity.location:
                        response += f"  📍 {activity.location}\n"
                    if activity.max_participants:
                        response += f"  👥 Максимум участников: {activity.max_participants}\n"
                    if activity.participants:
                        response += f"  👥 Участники: {', '.join(p.name for p in activity.participants)}\n"
                    if activity.tags:
                        response += f"  🏷️ Теги: {activity.tags}\n"
                    response += "\n"
            return response
        
        return "Активности не найдены."
    finally:
        session.close()

def search_general_info(query: str) -> str:
    """Search for general information based on the query."""
    query_lower = query.lower()
    
    # База знаний
    if 'база знаний' in query_lower or 'wiki' in query_lower:
        return (
            "📚 База знаний доступна по адресу: wiki.company.com\n\n"
            "Для доступа используйте ваши корпоративные учетные данные.\n"
            "Если у вас нет доступа, обратитесь к вашему руководителю или в IT-отдел."
        )
    
    # Офис
    if 'офис' in query_lower or 'находится' in query_lower:
        return (
            "🏢 Офис находится по адресу:\n"
            "г. Москва, ул. Примерная, д. 123\n\n"
            "Ближайшее метро: Примерная (5 минут пешком)\n"
            "Вход через главный вход, предъявите пропуск на ресепшене."
        )
    
    # Правила
    if 'правила' in query_lower or 'политика' in query_lower:
        return (
            "📋 Основные правила компании:\n\n"
            "1. Рабочий день с 9:00 до 18:00\n"
            "2. Обед с 13:00 до 14:00\n"
            "3. Дресс-код: business casual\n"
            "4. Обязательное использование корпоративной почты\n"
            "5. Соблюдение политики информационной безопасности\n\n"
            "Полные правила доступны в базе знаний."
        )
    
    # IT поддержка
    if 'it' in query_lower or 'поддержка' in query_lower or 'помощь' in query_lower:
        return (
            "🖥️ IT поддержка:\n\n"
            "• Email: support@company.com\n"
            "• Внутренний номер: 1234\n"
            "• Часы работы: 9:00 - 18:00\n\n"
            "Для срочных вопросов звоните на внутренний номер."
        )
    
    return "Информация не найдена."

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user messages and respond accordingly."""
    query = update.message.text
    logger.info(f"Received message: {query}")
    
    category, confidence = classify_query(query)
    logger.info(f"Classified as: {category} with confidence {confidence:.2f}")
    
    if category == "неопределенный запрос":
        # Пробуем найти ответ в общей информации
        response = search_general_info(query)
        if response == "Информация не найдена.":
            response = (
                "Извините, я не совсем понял ваш вопрос. Попробуйте переформулировать или используйте /help для получения подсказок.\n\n"
                "Примеры вопросов:\n"
                "• Кто работает в IT отделе?\n"
                "• Какие мероприятия запланированы на этой неделе?\n"
                "• Какие задачи у Ивана Петрова?\n"
                "• Кто хочет поиграть в настольные игры?\n"
                "• Как получить доступ к базе знаний?\n"
                "• Где находится офис?"
            )
    elif category == "поиск сотрудника":
        response = search_employees(query)
    elif category == "информация о мероприятии":
        response = search_events(query)
    elif category == "информация о задаче":
        response = search_tasks(query)
    elif category == "социальные активности":
        response = search_activities(query)
    elif category == "общая информация":
        response = search_general_info(query)
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
