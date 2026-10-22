@echo off
echo ============================================================
echo Starting LocMau IDH AI Prediction Microservice (FastAPI)...
echo Running on http://127.0.0.1:8001
echo ============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m uvicorn idh_ai_service.main:app --host 0.0.0.0 --port 8001 --reload
pause
