import pandas as pd

# Sample employee data
employees_data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['Иван Петров', 'Мария Сидорова', 'Алексей Иванов', 'Елена Смирнова', 'Дмитрий Козлов'],
    'department': ['IT', 'HR', 'Sales', 'Marketing', 'IT'],
    'project': ['Project A', 'Project B', 'Project C', 'Project A', 'Project B'],
    'position': ['Senior Developer', 'HR Manager', 'Sales Manager', 'Marketing Specialist', 'Developer'],
    'birthday': ['1990-05-15', '1988-08-23', '1992-03-10', '1995-11-30', '1991-07-20']
}

# Sample events data
events_data = {
    'id': [1, 2, 3, 4],
    'name': ['Корпоратив', 'Тренинг по продажам', 'Встреча с клиентом', 'Презентация проекта'],
    'date': ['2024-03-15', '2024-03-20', '2024-03-25', '2024-04-01'],
    'type': ['Корпоративное мероприятие', 'Обучение', 'Встреча', 'Презентация'],
    'description': ['Ежегодный корпоратив компании', 'Тренинг для отдела продаж', 'Встреча с ключевым клиентом', 'Презентация нового проекта']
}

# Sample tasks data
tasks_data = {
    'id': [1, 2, 3, 4],
    'title': ['Подготовить отчет', 'Создать презентацию', 'Провести встречу', 'Обновить документацию'],
    'deadline': ['2024-03-18', '2024-03-22', '2024-03-25', '2024-03-30'],
    'status': ['В процессе', 'Завершено', 'Новое', 'В процессе'],
    'assignee': ['Иван Петров', 'Мария Сидорова', 'Алексей Иванов', 'Дмитрий Козлов']
}

# Create DataFrames
employees_df = pd.DataFrame(employees_data)
events_df = pd.DataFrame(events_data)
tasks_df = pd.DataFrame(tasks_data)

def get_employees():
    return employees_df

def get_events():
    return events_df

def get_tasks():
    return tasks_df 