from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Table, Enum, Text, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
from typing import Optional
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

# Create base class for declarative models
Base = declarative_base()

# Create engine
engine = create_engine(os.getenv('DATABASE_URL', 'sqlite:///corporate_bot.db'))

# Create session factory
Session = sessionmaker(bind=engine)

def get_session():
    return Session()

class TaskStatus(enum.Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"

class EventType(enum.Enum):
    MEETING = "meeting"
    TRAINING = "training"
    TEAM_BUILDING = "team_building"
    PRESENTATION = "presentation"
    OTHER = "other"

class ActivityType(enum.Enum):
    SPORTS = "sports"
    GAMES = "games"
    LEARNING = "learning"
    SOCIAL = "social"
    OTHER = "other"

# Association tables
activity_participants = Table(
    'activity_participants',
    Base.metadata,
    Column('activity_id', Integer, ForeignKey('activities.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

event_participants = Table(
    'event_participants',
    Base.metadata,
    Column('event_id', Integer, ForeignKey('events.id')),
    Column('employee_id', Integer, ForeignKey('employees.id'))
)

class Employee(Base):
    __tablename__ = 'employees'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    surname = Column(String(100), nullable=False)
    position = Column(String(100), nullable=False)
    department = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    phone = Column(String(20))
    skills = Column(Text)
    interests = Column(Text)
    birthday = Column(DateTime)
    hire_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    timezone = Column(String(50), default='UTC')
    preferred_language = Column(String(10), default='ru')
    avatar_url = Column(String(200))
    bio = Column(Text)
    social_links = Column(Text)  # JSON string of social media links
    
    # Relationships
    assigned_tasks = relationship("Task", foreign_keys="Task.assignee_id", back_populates="assignee")
    created_tasks = relationship("Task", foreign_keys="Task.creator_id", back_populates="creator")
    events = relationship("Event", secondary=event_participants, back_populates="participants")
    activities = relationship("Activity", secondary=activity_participants, back_populates="participants")
    
    def __repr__(self):
        return f"<Employee {self.name} {self.surname}>"

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    location = Column(String(200))
    event_type = Column(Enum(EventType), nullable=False)
    organizer_id = Column(Integer, ForeignKey('employees.id'))
    max_participants = Column(Integer)
    is_online = Column(Boolean, default=False)
    meeting_link = Column(String(200))
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organizer = relationship("Employee", foreign_keys=[organizer_id])
    participants = relationship("Employee", secondary="event_participants", back_populates="events")
    
    def __repr__(self):
        return f"<Event {self.title}>"

class Task(Base):
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    status = Column(Enum(TaskStatus), default=TaskStatus.TODO)
    priority = Column(Integer, default=0)
    assignee_id = Column(Integer, ForeignKey('employees.id'))
    creator_id = Column(Integer, ForeignKey('employees.id'))
    due_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    estimated_hours = Column(Float)
    actual_hours = Column(Float)
    tags = Column(Text)  # JSON string of tags
    attachments = Column(Text)  # JSON string of attachment URLs
    
    # Relationships
    assignee = relationship("Employee", foreign_keys=[assignee_id], back_populates="assigned_tasks")
    creator = relationship("Employee", foreign_keys=[creator_id], back_populates="created_tasks")
    
    def __repr__(self):
        return f"<Task {self.title}>"

class Activity(Base):
    __tablename__ = 'activities'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    activity_type = Column(Enum(ActivityType), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    location = Column(String(200))
    organizer_id = Column(Integer, ForeignKey('employees.id'))
    max_participants = Column(Integer)
    current_participants = Column(Integer, default=0)
    status = Column(String(20), default='active')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organizer = relationship("Employee", foreign_keys=[organizer_id])
    participants = relationship("Employee", secondary=activity_participants, back_populates="activities")
    
    def __repr__(self):
        return f"<Activity {self.title}>"

class GeneralInfo(Base):
    __tablename__ = 'general_info'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50))
    tags = Column(Text)  # JSON string of tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<GeneralInfo {self.title}>"

# Create all tables
def init_db():
    Base.metadata.create_all(engine)

def parse_date(date_str):
    """Parse date string to datetime object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def parse_time(time_str):
    """Parse time string to time object."""
    return datetime.strptime(time_str, "%H:%M").time()
