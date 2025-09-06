$ErrorActionPreference = "Stop"

if (!(Test-Path -Path ".venv")) {
    Write-Host "Virtual environment not found. Running bootstrap..."
    & powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
}

Write-Host "Launching tiny task..."
& .\.venv\Scripts\Activate.ps1
py app.py

