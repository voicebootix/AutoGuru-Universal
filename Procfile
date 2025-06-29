web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
worker: celery -A backend.tasks.content_generation worker --loglevel=info 