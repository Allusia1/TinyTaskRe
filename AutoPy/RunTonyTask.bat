@echo off
setlocal ENABLEDELAYEDEXPANSION
pushd "%~dp0"

REM Ensure virtual environment exists
if not exist ".venv\Scripts\python.exe" (
  echo [TonyTask] Creating virtual environment...
  py -m venv .venv
)

set "PYEXE=%~dp0.venv\Scripts\python.exe"
set "PYWEXE=%~dp0.venv\Scripts\pythonw.exe"

REM Install/upgrade requirements
"%PYEXE%" -m pip install --disable-pip-version-check --upgrade pip
"%PYEXE%" -m pip install -r requirements.txt

REM Launch the app without a console window
start "TonyTask" "%PYWEXE%" "%~dp0app.py"

popd
endlocal

