from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Enum, Boolean, Table, Text, DateTime
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import enum
import os
from datetime import datetime, timedelta

# Create the base class using the new recommended approach
Base = declarative_base()

# Create engine and session
engine = create_engine('sqlite:///corporate.db')
Session = sessionmaker(bind=engine)

class TaskStatus(enum.Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    BLOCKED = "Blocked"
    REVIEW = "Review"

class ActivityType(enum.Enum):
    GAME = "Game"
    LUNCH = "Lunch"
    MEETING = "Meeting"
    TRAINING = "Training"
    TEAM_BUILDING = "Team Building"
    OTHER = "Other"

class EventType(enum.Enum):
    CORPORATE = "Corporate"
    TRAINING = "Training"
    MEETING = "Meeting"
    BIRTHDAY = "Birthday"
    HOLIDAY = "Holiday"
    CONFERENCE = "Conference"
    OTHER = "Other"

# Association tables
activity_participants = Table('activity_participants', Base.metadata,
    Column('activity_id', Integer, ForeignKey('activities.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

event_participants = Table('event_participants', Base.metadata,
    Column('event_id', Integer, ForeignKey('events.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)
    department = Column(String, nullable=False)
    project = Column(String)
    hire_date = Column(Date, nullable=False)
    birthday = Column(Date)
    interests = Column(String)  # Comma-separated list of interests
    email = Column(String)
    phone = Column(String)
    skills = Column(String)  # Comma-separated list of skills
    bio = Column(Text)
    
    tasks = relationship("Task", back_populates="assignee")
    activities = relationship("Activity", secondary=activity_participants, back_populates="participants")
    events = relationship("Event", secondary=event_participants, back_populates="participants")
    created_activities = relationship("Activity", back_populates="creator")
    organized_events = relationship("Event", back_populates="organizer")

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(Enum(EventType), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String)  # Format: "HH:MM"
    description = Column(Text)
    location = Column(String)
    organizer_id = Column(Integer, ForeignKey('employees.id'))
    is_recurring = Column(Boolean, default=False)
    recurrence_pattern = Column(String)  # e.g., "weekly", "monthly"
    
    organizer = relationship("Employee", back_populates="organized_events")
    participants = relationship("Employee", secondary=event_participants, back_populates="events")

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    deadline = Column(Date, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False)
    assignee_id = Column(Integer, ForeignKey('employees.id'))
    priority = Column(Integer, default=1)  # 1-5, where 5 is highest
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    tags = Column(String)  # Comma-separated list of tags
    
    assignee = relationship("Employee", back_populates="tasks")

class Activity(Base):
    __tablename__ = 'activities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(Enum(ActivityType), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String)  # Format: "HH:MM"
    description = Column(Text)
    location = Column(String)
    max_participants = Column(Integer)
    is_active = Column(Boolean, default=True)
    creator_id = Column(Integer, ForeignKey('employees.id'))
    created_at = Column(DateTime, default=datetime.now)
    tags = Column(String)  # Comma-separated list of tags
    
    creator = relationship("Employee", back_populates="created_activities")
    participants = relationship("Employee", secondary=activity_participants, back_populates="activities")

def parse_date(date_str):
    """Преобразует строку даты в объект datetime.date"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()

def init_db():
    """Initialize the database with test data."""
    # Drop all tables and recreate them
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    session = Session()
    
    try:
        # Create test employees
        employees = [
            Employee(
                name="Иван Петров",
                position="Senior Developer",
                department="IT",
                email="ivan@company.com",
                phone="+7 (999) 123-45-67",
                hire_date=parse_date("2020-01-15"),
                birthday=parse_date("1985-05-20"),
                skills="Python, Django, PostgreSQL, Docker",
                interests="настольные игры, программирование, путешествия",
                bio="Опытный разработчик с 10-летним стажем"
            ),
            Employee(
                name="Анна Сидорова",
                position="HR Manager",
                department="HR",
                email="anna@company.com",
                phone="+7 (999) 234-56-78",
                hire_date=parse_date("2019-03-10"),
                birthday=parse_date("1990-08-15"),
                skills="HR, рекрутинг, обучение персонала",
                interests="йога, танцы, психология",
                bio="HR специалист с опытом в IT компаниях"
            ),
            Employee(
                name="Дмитрий Козлов",
                position="Developer",
                department="IT",
                email="dmitry@company.com",
                phone="+7 (999) 345-67-89",
                hire_date=parse_date("2021-06-01"),
                birthday=parse_date("1995-03-25"),
                skills="Python, FastAPI, MongoDB, React",
                interests="настольные игры, спорт, музыка",
                bio="Full-stack разработчик"
            ),
            Employee(
                name="Мария Иванова",
                position="QA Engineer",
                department="IT",
                email="maria@company.com",
                phone="+7 (999) 456-78-90",
                hire_date=parse_date("2022-02-15"),
                birthday=parse_date("1992-11-10"),
                skills="Python, Selenium, Pytest, Postman",
                interests="тестирование, танцы, йога, путешествия",
                bio="QA инженер с опытом автоматизации тестирования"
            ),
            Employee(
                name="Алексей Смирнов",
                position="Project Manager",
                department="IT",
                email="alexey@company.com",
                phone="+7 (999) 567-89-01",
                hire_date=parse_date("2018-09-01"),
                birthday=parse_date("1988-07-05"),
                skills="Agile, Scrum, Jira, Python",
                interests="настольные игры, теннис, чтение",
                bio="Опытный проект-менеджер в IT"
            )
        ]
        
        # Add employees to session
        session.add_all(employees)
        session.flush()  # Flush to get IDs
        
        # Create test events
        events = [
            Event(
                name="Python Meetup",
                type=EventType.CONFERENCE,
                date=parse_date("2025-05-20"),
                time="15:00",
                location="Конференц-зал",
                description="Встреча Python-разработчиков компании",
                participants=employees[:3]  # Иван, Анна, Дмитрий
            ),
            Event(
                name="Тренинг по Agile",
                type=EventType.TRAINING,
                date=parse_date("2025-05-22"),
                time="10:00",
                location="Тренинг-зал",
                description="Обучение методологии Agile",
                participants=employees[1:]  # Все кроме Ивана
            ),
            Event(
                name="День рождения Анны",
                type=EventType.BIRTHDAY,
                date=parse_date("2025-08-15"),
                time="12:00",
                location="Офис",
                description="Празднование дня рождения",
                participants=employees
            )
        ]
        
        # Add events to session
        session.add_all(events)
        session.flush()
        
        # Create test tasks
        tasks = [
            Task(
                title="Рефакторинг API",
                description="Оптимизация существующего API",
                status=TaskStatus.IN_PROGRESS,
                deadline=parse_date("2025-05-25"),
                assignee=employees[0],  # Иван
                tags="python, api, optimization"
            ),
            Task(
                title="Написание тестов",
                description="Автоматизация тестирования",
                status=TaskStatus.TODO,
                deadline=parse_date("2025-05-30"),
                assignee=employees[3],  # Мария
                tags="testing, automation, python"
            ),
            Task(
                title="Исправление бага в авторизации",
                description="Критический баг в системе авторизации",
                status=TaskStatus.BLOCKED,
                deadline=parse_date("2025-05-18"),
                assignee=employees[2],  # Дмитрий
                tags="bug, auth, critical"
            )
        ]
        
        # Add tasks to session
        session.add_all(tasks)
        session.flush()
        
        # Create test activities
        activities = [
            Activity(
                name="Настольные игры",
                type=ActivityType.GAME,
                date=parse_date("2025-05-21"),
                time="18:00",
                location="Игровая комната",
                description="Еженедельные настольные игры",
                max_participants=8,
                is_active=True,
                participants=employees[:3],  # Иван, Анна, Дмитрий
                tags="games, team building"
            ),
            Activity(
                name="Йога в офисе",
                type=ActivityType.TRAINING,
                date=parse_date("2025-05-23"),
                time="09:00",
                location="Тренинг-зал",
                description="Утренняя йога для сотрудников",
                max_participants=10,
                is_active=True,
                participants=[employees[1], employees[3]],  # Анна, Мария
                tags="yoga, health, morning"
            ),
            Activity(
                name="Совместный обед",
                type=ActivityType.LUNCH,
                date=parse_date("2025-05-24"),
                time="13:00",
                location="Столовая",
                description="Еженедельный обед команды",
                max_participants=6,
                is_active=True,
                participants=employees,
                tags="lunch, team building"
            )
        ]
        
        # Add activities to session
        session.add_all(activities)
        
        # Commit all changes
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_session():
    """Get a new database session."""
    return Session() 
