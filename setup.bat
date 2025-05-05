@echo off
echo ===================================================
echo Jigsaw Unintended Bias Audit - Environment Setup
echo ===================================================
echo.

REM Check Python installation
python --version 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python not found! Please install Python 3.8+ and try again.
    exit /b
)

echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Environment setup complete!
echo.
echo To use this environment:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run setup_environment.py to download the dataset: python setup_environment.py --kaggle_json path\to\kaggle.json
echo 3. Start Jupyter notebook: jupyter notebook
echo.
echo ===================================================

pause 