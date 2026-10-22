Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting LocMau IDH AI Prediction Microservice (FastAPI)..." -ForegroundColor Green
Write-Host "Running on http://127.0.0.1:8001" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

& "$scriptDir\.venv\Scripts\python.exe" -m uvicorn idh_ai_service.main:app --host 0.0.0.0 --port 8001 --reload
