from flask import Flask, render_template, request, jsonify
from models import get_session, Employee, Event, Task, Activity
from datetime import datetime, timedelta
from transformers import pipeline
import torch
import json
from telegram_bot import analyze_query, classify_query

app = Flask(__name__)

# Initialize the AI model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    top_k=3
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Analyze the query using the same AI model as the bot
    category, confidence = classify_query(query)
    
    session = get_session()
    try:
        if category == "поиск сотрудника":
            results = search_employees(session, query)
        elif category == "информация о мероприятии":
            results = search_events(session, query)
        elif category == "информация о задаче":
            results = search_tasks(session, query)
        elif category == "социальные активности":
            results = search_activities(session, query)
        else:
            results = search_general_info(session, query)
        
        return jsonify({
            'category': category,
            'confidence': confidence,
            'results': results
        })
    finally:
        session.close()

def search_employees(session, query):
    employees = session.query(Employee).all()
    results = []
    for emp in employees:
        if any(term.lower() in emp.name.lower() or 
               term.lower() in emp.department.lower() or 
               term.lower() in emp.skills.lower() or 
               term.lower() in emp.interests.lower() 
               for term in query.split()):
            results.append({
                'name': emp.name,
                'position': emp.position,
                'department': emp.department,
                'skills': emp.skills,
                'interests': emp.interests
            })
    return results

def search_events(session, query):
    events = session.query(Event).all()
    results = []
    for event in events:
        if any(term.lower() in event.name.lower() or 
               term.lower() in event.description.lower() 
               for term in query.split()):
            results.append({
                'name': event.name,
                'type': event.type.value,
                'date': event.date.strftime('%Y-%m-%d'),
                'time': event.time.strftime('%H:%M'),
                'location': event.location,
                'description': event.description
            })
    return results

def search_tasks(session, query):
    tasks = session.query(Task).all()
    results = []
    for task in tasks:
        if any(term.lower() in task.title.lower() or 
               term.lower() in task.description.lower() 
               for term in query.split()):
            results.append({
                'title': task.title,
                'description': task.description,
                'status': task.status.value,
                'priority': task.priority,
                'deadline': task.deadline.strftime('%Y-%m-%d') if task.deadline else None
            })
    return results

def search_activities(session, query):
    activities = session.query(Activity).filter(Activity.is_active == True).all()
    results = []
    for activity in activities:
        if any(term.lower() in activity.name.lower() or 
               term.lower() in activity.description.lower() 
               for term in query.split()):
            results.append({
                'name': activity.name,
                'type': activity.type.value,
                'date': activity.date.strftime('%Y-%m-%d'),
                'time': activity.time.strftime('%H:%M'),
                'location': activity.location,
                'description': activity.description,
                'max_participants': activity.max_participants
            })
    return results

def search_general_info(session, query):
    # Combine results from all categories
    results = {
        'employees': search_employees(session, query),
        'events': search_events(session, query),
        'tasks': search_tasks(session, query),
        'activities': search_activities(session, query)
    }
    return results

if __name__ == '__main__':
    app.run(debug=True) 