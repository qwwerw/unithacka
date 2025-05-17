from sqlalchemy import create_engine, Column, Integer, String, Date, Time, DateTime, Boolean, ForeignKey, Enum, Text, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, time, timedelta
import enum
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create database engine
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///corporate_bot.db'))
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Enums
class EventType(enum.Enum):
    MEETING = "meeting"
    TRAINING = "training"
    HOLIDAY = "holiday"
    OTHER = "other"

class ActivityType(enum.Enum):
    YOGA = "yoga"
    SPORT = "sport"
    GAMES = "games"
    OTHER = "other"

class TaskStatus(enum.Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"

# Association tables
event_participants = Table('event_participants', Base.metadata,
    Column('event_id', Integer, ForeignKey('events.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

activity_participants = Table('activity_participants', Base.metadata,
    Column('activity_id', Integer, ForeignKey('activities.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

# Models
class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    position = Column(String(100))
    department = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    hire_date = Column(Date)
    birthday = Column(Date)
    skills = Column(String(500))
    interests = Column(String(500))
    bio = Column(Text)
    
    # Relationships
    tasks = relationship("Task", back_populates="assignee")
    events = relationship("Event", secondary=event_participants, back_populates="participants")
    activities = relationship("Activity", secondary=activity_participants, back_populates="participants")

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(String(1000))
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    location = Column(String(200))
    type = Column(Enum(EventType), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    participants = relationship("Employee", secondary=event_participants, back_populates="events")

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(String(1000))
    deadline = Column(Date)
    status = Column(Enum(TaskStatus), default=TaskStatus.NEW)
    priority = Column(String(50))
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Foreign keys
    assignee_id = Column(Integer, ForeignKey('employees.id'))
    
    # Relationships
    assignee = relationship("Employee", back_populates="tasks")

class Activity(Base):
    __tablename__ = 'activities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000))
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    location = Column(String(200))
    type = Column(Enum(ActivityType), nullable=False)
    max_participants = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    participants = relationship("Employee", secondary=activity_participants, back_populates="activities")

class GeneralInfo(Base):
    __tablename__ = 'general_info'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100))
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def get_session():
    """Get a new database session."""
    return Session()

def parse_date(date_str):
    """Parse date string to datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def parse_time(time_str):
    """Parse time string to time object."""
    return datetime.strptime(time_str, "%H:%M").time()
