@echo off
echo Starting AutoGuru Universal...
docker-compose up -d
timeout /t 10
pip install -r requirements.txt
python -c "import asyncio; from backend.database.connection import init_database; asyncio.run(init_database())"
start /b uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo Open http://localhost:8000/docs to see your API!
pause 