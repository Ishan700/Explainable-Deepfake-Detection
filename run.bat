@echo off
rem Double-click to start the web app and open it in your browser.
rem Needs the one-time setup from the README first (venv + pip install).
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo No virtual environment found. Run the setup from the README first:
    echo    python -m venv .venv
    echo    .venv\Scripts\activate
    echo    pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo Starting the server... the browser opens in a few seconds.
echo Close this window to stop the server.
rem give the model ~8 s to load before the browser opens
start "" cmd /c "timeout /t 8 /nobreak >nul && start "" http://127.0.0.1:5000"
python app.py
pause
