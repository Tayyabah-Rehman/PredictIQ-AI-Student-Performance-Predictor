@echo off
echo ============================================================
echo   AI Student Performance Predictor v2 - Setup
echo ============================================================
py -3.10 --version >nul 2>&1
if %errorlevel% NEQ 0 (echo [ERROR] Python 3.10 not found. & pause & exit /b 1)
echo [OK] Python 3.10 found
if not exist ".venv" ( echo [+] Creating virtual environment... & py -3.10 -m venv .venv )
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo [+] Training model (~60 seconds)...
python model/train_model.py
echo [+] Generating dashboard charts (~10 seconds)...
python dashboard.py
echo.
echo ============================================================
echo  DONE! Run:  python app.py
echo  Predictor:  http://localhost:5000
echo  Dashboard:  http://localhost:5000/dashboard
echo ============================================================
pause
