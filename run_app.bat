@echo off
TITLE Local GenAI Forge - Runtime
CLS

ECHO ========================================================
ECHO   LOCAL GENAI FORGE | LAUNCHER
ECHO ========================================================
ECHO.

:: 1. Environment Check
:: Before trying to run, we make sure the user actually ran the setup script.
IF NOT EXIST ".venv" (
    ECHO [!] Critical Error: Virtual environment (.venv) not found.
    ECHO [!] Action Required: Please run 'setup_windows.bat' first to install dependencies.
    ECHO.
    PAUSE
    EXIT /B
)

:: 2. Launch Sequence
ECHO [*] Initializing inference pipeline...
ECHO [*] Dashboard will open in your default browser automatically.
ECHO.
ECHO [INFO] Keep this terminal window OPEN. It displays backend logs and telemetry.
ECHO.

:: 3. Execution
:: I'm calling Streamlit via the venv python executable directly.
:: This is safer than 'activating' the venv in script, as it prevents PATH conflicts.
.venv\Scripts\python -m streamlit run ui/app.py

:: 4. Exit Handling
:: If the app crashes, pause so the user can read the traceback error.
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [!] The application crashed. See the error above.
    PAUSE
)