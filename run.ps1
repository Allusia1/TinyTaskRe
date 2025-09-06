$ErrorActionPreference = "Stop"

Write-Host "[tiny task] Preparing environment..."

if (!(Test-Path -Path ".venv")) {
    Write-Host "[tiny task] Creating virtual environment (.venv)..."
    py -m venv .venv
}

$python = Join-Path (Resolve-Path ".venv") "Scripts\python.exe"

Write-Host "[tiny task] Upgrading pip and installing requirements..."
& $python -m pip install --upgrade pip | Out-Null
& $python -m pip install -r requirements.txt | Out-Null

Write-Host "[tiny task] Launching app..."
& $python app.py

