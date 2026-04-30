@echo off
chcp 65001 >nul
echo ============================================
echo   VoiceProject Setup
echo ============================================
echo.

echo [1/6] Current path: %CD%
echo.

echo [2/6] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.11: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install!
    pause
    exit /b 1
)
python --version
echo.

echo [3/6] Creating virtual environment (.venv)...
if exist .venv (
    echo Virtual environment already exists. Skipping.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo Done!
)
echo.

echo [4/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

echo [5/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [6/6] Installing packages...
echo   Step 1: Installing pre-built av wheel (no C++ compiler needed)...
pip install av==12.3.0 --only-binary=:all:

echo   Step 2: Installing remaining packages...
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
echo.

echo ============================================
echo   Setup complete!
echo   To activate in VSCode terminal type:
echo   .venv\Scripts\activate
echo ============================================
pause
