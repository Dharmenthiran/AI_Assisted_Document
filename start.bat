@echo off
title Document AI Assistant - RAG
echo ========================================================
echo        Starting Document AI Assistant (Local RAG)
echo ========================================================
echo.

cd /d "%~dp0"

REM Check if .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv.
    echo Please run: py -3.11 -m venv .venv
    echo and: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if .env exists
if not exist ".env" (
    echo [.env file missing] Copying from .env.example...
    copy .env.example .env
)

echo Activating virtual environment and starting server on http://127.0.0.1:8000...
echo (Make sure Ollama is running in the background: 'ollama serve')
echo.
.venv\Scripts\python.exe run.py
pause
