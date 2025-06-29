@echo off
echo Starting AutoGuru Universal...
echo.
echo 1. Setting environment...
if not exist .env (
    copy env_template.txt .env
    echo Created .env file from template
)

echo.
echo 2. Installing dependencies...
pip install -r requirements.txt

echo.
echo 3. Starting the application...
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

pause
