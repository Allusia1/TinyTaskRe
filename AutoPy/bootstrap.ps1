$ErrorActionPreference = "Stop"

Write-Host "Setting up virtual environment..."

if (!(Test-Path -Path ".venv")) {
    py -m venv .venv
}

Write-Host "Activating virtual environment and installing requirements..."

& .\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

Write-Host "Bootstrap complete. Use launch.ps1 to start the app."

