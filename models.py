from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import enum
from datetime import datetime

Base = declarative_base()

class TaskStatus(enum.Enum):
    NEW = "Новое"
    IN_PROGRESS = "В процессе"
    COMPLETED = "Завершено"

class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False)
    project = Column(String, nullable=False)
    position = Column(String, nullable=False)
    birthday = Column(Date, nullable=False)

    tasks = relationship("Task", back_populates="assignee")

class Event(Base):
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    type = Column(String, nullable=False)
    description = Column(String, nullable=False)

class Task(Base):
    __tablename__ = 'tasks'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    deadline = Column(Date, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False)
    assignee_id = Column(Integer, ForeignKey('employees.id'))
    
    assignee = relationship("Employee", back_populates="tasks")

# Создание подключения к базе данных
engine = create_engine('sqlite:///corporate.db')
Session = sessionmaker(bind=engine)

def parse_date(date_str):
    """Преобразует строку даты в объект datetime.date"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()

def init_db():
    """Инициализация базы данных"""
    Base.metadata.create_all(engine)
    
    # Создаем сессию
    session = Session()
    
    # Проверяем, есть ли уже данные
    if session.query(Employee).count() == 0:
        # Добавляем тестовые данные
        employees = [
            Employee(
                name='Иван Петров',
                department='IT',
                project='Project A',
                position='Senior Developer',
                birthday=parse_date('1990-05-15')
            ),
            Employee(
                name='Мария Сидорова',
                department='HR',
                project='Project B',
                position='HR Manager',
                birthday=parse_date('1988-08-23')
            ),
            Employee(
                name='Алексей Иванов',
                department='Sales',
                project='Project C',
                position='Sales Manager',
                birthday=parse_date('1992-03-10')
            ),
            Employee(
                name='Елена Смирнова',
                department='Marketing',
                project='Project A',
                position='Marketing Specialist',
                birthday=parse_date('1995-11-30')
            ),
            Employee(
                name='Дмитрий Козлов',
                department='IT',
                project='Project B',
                position='Developer',
                birthday=parse_date('1991-07-20')
            )
        ]
        
        events = [
            Event(
                name='Корпоратив',
                date=parse_date('2024-03-15'),
                type='Корпоративное мероприятие',
                description='Ежегодный корпоратив компании'
            ),
            Event(
                name='Тренинг по продажам',
                date=parse_date('2024-03-20'),
                type='Обучение',
                description='Тренинг для отдела продаж'
            ),
            Event(
                name='Встреча с клиентом',
                date=parse_date('2024-03-25'),
                type='Встреча',
                description='Встреча с ключевым клиентом'
            ),
            Event(
                name='Презентация проекта',
                date=parse_date('2024-04-01'),
                type='Презентация',
                description='Презентация нового проекта'
            )
        ]
        
        tasks = [
            Task(
                title='Подготовить отчет',
                deadline=parse_date('2024-03-18'),
                status=TaskStatus.IN_PROGRESS,
                assignee_id=1
            ),
            Task(
                title='Создать презентацию',
                deadline=parse_date('2024-03-22'),
                status=TaskStatus.COMPLETED,
                assignee_id=2
            ),
            Task(
                title='Провести встречу',
                deadline=parse_date('2024-03-25'),
                status=TaskStatus.NEW,
                assignee_id=3
            ),
            Task(
                title='Обновить документацию',
                deadline=parse_date('2024-03-30'),
                status=TaskStatus.IN_PROGRESS,
                assignee_id=5
            )
        ]
        
        # Добавляем все данные в сессию
        session.add_all(employees)
        session.add_all(events)
        session.add_all(tasks)
        
        # Сохраняем изменения
        session.commit()
    
    session.close()

def get_session():
    """Получение сессии базы данных"""
    return Session() 