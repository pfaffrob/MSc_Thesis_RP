import requests
import json
from datetime import datetime
from IPython.display import display, HTML

def fetch_todoist_tasks(labels=["meeting"], created_after="2025-03-01", created_before="2025-03-31"):
    LOG_FILE = "task_log.json"

    # Reset variables to avoid retaining values from previous executions
    tasks_html = ""
    current_tasks = {}
    completed_tasks = []

    # Read API token from file
    with open('../../../.obsidian/todoist-token', 'r') as file:
        API_TOKEN = file.read().strip()

    # Define API URL and query parameters
    url = "https://api.todoist.com/rest/v2/tasks"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    # Load previous tasks from log file
    try:
        with open(LOG_FILE, "r") as f:
            logged_tasks = json.load(f)
    except FileNotFoundError:
        logged_tasks = {}

    # Fetch current tasks
    response = requests.get(url, headers=headers)
    tasks_html = ""

    if response.status_code == 200:
        tasks = response.json()
        current_tasks = {}
        completed_tasks = logged_tasks.get("completed", [])
        
        for task in tasks:
            task_id = str(task.get("id"))
            created_date = task.get("created_at", "")
            task_labels = task.get("labels", [])
            content = task.get("content", "")
            
            if created_date:
                task_date = datetime.fromisoformat(created_date[:-1])
                if (not labels or any(label in task_labels for label in labels)) and \
                   (not created_after or task_date >= datetime.fromisoformat(created_after)) and \
                   (not created_before or task_date <= datetime.fromisoformat(created_before)):
                    current_tasks[task_id] = {
                        "content": content,
                        "created_at": created_date,
                        "labels": task_labels
                    }

        
        # Determine newly completed tasks
        previous_task_ids = set(logged_tasks.get("current", {}).keys())
        current_task_ids = set(current_tasks.keys())
        newly_completed = previous_task_ids - current_task_ids
        
        for task_id in newly_completed:
            completed_tasks.append(logged_tasks["current"][task_id])
        
        # Save updated tasks to log file
        with open(LOG_FILE, "w") as f:
            json.dump({"current": current_tasks, "completed": completed_tasks}, f, indent=4)
        
        # Build HTML output
        tasks_html = "<ul class='list-unstyled'>"
        
        for task_id, task_data in current_tasks.items():
            labels_html = " ".join([f"<span style='background-color: #555; color: white; padding: 2px 5px; border-radius: 3px;'>#{label}</span>" for label in task_data['labels'] if label != "meeting"])
            label_text = f"<br><br>{labels_html}" if labels_html else ""
            tasks_html += f"<li class='list-group-item' style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #343a40; color: white;'><strong>{task_data['content']}</strong>{label_text}</li>"
        
        for task_data in completed_tasks:
            labels_html = " ".join([f"<span style='background-color: #555; color: white; padding: 2px 5px; border-radius: 3px;'>#{label}</span>" for label in task_data.get('labels', []) if label != "meeting"])
            label_text = f"<br><br>{labels_html}" if labels_html else ""
            tasks_html += f"<li class='list-group-item' style='margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #28a745; color: white;'>{task_data['content']}{label_text}</li>"
        
        tasks_html += "</ul>"
        
        if not current_tasks and not completed_tasks:
            tasks_html = "<p>No tasks found matching the criteria.</p>"
    else:
        tasks_html = f"<p>Failed to fetch tasks: {response.status_code}</p>"

    # Output the HTML directly
    display(HTML(tasks_html))