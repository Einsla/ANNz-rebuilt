@echo off
setlocal
cd /d "%~dp0"

echo Starting the ANNzPlus Python app...
python -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo The app could not start. Check that Python and the project requirements are installed.
    echo You can install dependencies with: python -m pip install -r requirements.txt
    echo.
    pause
)
endlocal
