from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
import enum
import os
from datetime import datetime

# Create the base class using the new recommended approach
Base = declarative_base()

# Create engine and session
engine = create_engine('sqlite:///corporate.db')
Session = sessionmaker(bind=engine)

class TaskStatus(enum.Enum):
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)
    department = Column(String, nullable=False)
    project = Column(String)
    hire_date = Column(Date, nullable=False)
    
    tasks = relationship("Task", back_populates="assignee")

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    description = Column(String)

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String)
    deadline = Column(Date, nullable=False)
    status = Column(Enum(TaskStatus), nullable=False)
    assignee_id = Column(Integer, ForeignKey('employees.id'))
    
    assignee = relationship("Employee", back_populates="tasks")

def parse_date(date_str):
    """Преобразует строку даты в объект datetime.date"""
    return datetime.strptime(date_str, '%Y-%m-%d').date()

def init_db():
    """Initialize the database with test data."""
    Base.metadata.create_all(engine)
    
    session = Session()
    
    # Check if data already exists
    if session.query(Employee).first():
        session.close()
        return
    
    # Create test employees
    employees = [
        Employee(
            name="Иван Петров",
            position="Senior Developer",
            department="IT",
            project="Project A",
            hire_date=parse_date("2023-01-15")
        ),
        Employee(
            name="Дмитрий Козлов",
            position="Developer",
            department="IT",
            project="Project B",
            hire_date=parse_date("2023-03-20")
        ),
        Employee(
            name="Анна Сидорова",
            position="HR Manager",
            department="HR",
            project=None,
            hire_date=parse_date("2023-02-01")
        )
    ]
    
    # Create test events
    events = [
        Event(
            name="Корпоративная вечеринка",
            type="Корпоратив",
            date=parse_date("2023-12-25"),
            description="Ежегодная корпоративная вечеринка"
        ),
        Event(
            name="Тренинг по Python",
            type="Обучение",
            date=parse_date("2023-11-15"),
            description="Тренинг по основам Python для разработчиков"
        )
    ]
    
    # Create test tasks
    tasks = [
        Task(
            title="Рефакторинг кода",
            description="Улучшение структуры существующего кода",
            deadline=parse_date("2023-12-01"),
            status=TaskStatus.IN_PROGRESS,
            assignee=employees[0]
        ),
        Task(
            title="Написание документации",
            description="Создание технической документации",
            deadline=parse_date("2023-11-30"),
            status=TaskStatus.TODO,
            assignee=employees[1]
        )
    ]
    
    # Add all objects to session and commit
    session.add_all(employees)
    session.add_all(events)
    session.add_all(tasks)
    session.commit()
    session.close()

def get_session():
    """Get a new database session."""
    return Session() 
