@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "ANNZ_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
    set "ANNZ_PYTHON=python"
)

echo Starting the ANNzPlus Python app...
"%ANNZ_PYTHON%" -m streamlit run "%~dp0app.py"

if errorlevel 1 (
    echo.
    echo The app could not start.
    if not exist ".venv\Scripts\python.exe" (
        echo Run Install_ANNzPlus.bat first to create the environment and install dependencies.
    ) else (
        echo Check the Python dependencies in the .venv environment.
    )
    echo.
    pause
)
endlocal
