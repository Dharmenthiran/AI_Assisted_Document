# Document AI Assistant PowerShell Launcher
$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting Document AI Assistant (Local RAG)       " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if .venv exists
$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at .venv." -ForegroundColor Red
    Write-Host "Please create it using: py -3.11 -m venv .venv" -ForegroundColor Yellow
    Write-Host "Then install dependencies: .venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# Check if .env exists
$envFile = Join-Path $scriptDir ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating .env from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
}

# Check Ollama connectivity
try {
    $res = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($res) {
        Write-Host "Ollama service is ONLINE." -ForegroundColor Green
    }
} catch {
    Write-Host "[WARNING] Ollama does not seem to be running on http://localhost:11434." -ForegroundColor Yellow
    Write-Host "Remember to run 'ollama serve' in another terminal." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting server at http://127.0.0.1:8000 ..." -ForegroundColor Green
Write-Host "Press Ctrl + C to stop the server." -ForegroundColor Gray
Write-Host ""

& $pythonExe run.py
