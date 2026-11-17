@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo        ANNzPlus Python - Windows Setup
echo ============================================
echo.

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_LAUNCHER=py -3"
        goto python_ready
    )
)

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.9 or newer was not found.
    echo Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
if errorlevel 1 (
    echo Python 3.9 or newer is required.
    pause
    exit /b 1
)
set "PYTHON_LAUNCHER=python"

:python_ready
if not exist ".venv\Scripts\python.exe" (
    echo Creating the isolated Python environment...
    %PYTHON_LAUNCHER% -m venv ".venv"
    if errorlevel 1 goto install_failed
)

echo Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_failed

echo Installing PyTorch and ANNzPlus dependencies.
echo This may take several minutes and requires an internet connection.
".venv\Scripts\python.exe" install.py
if errorlevel 1 goto install_failed

echo.
echo Creating desktop and Start Menu shortcuts...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Create_ANNzPlus_Shortcut.ps1"
if errorlevel 1 (
    echo The app was installed, but shortcut creation failed.
    echo You can still launch it using Run_ANNzPlus.bat.
)

echo.
echo Installation completed successfully.
echo Double-click Run_ANNzPlus.bat or use the ANNzPlus desktop shortcut.
pause
exit /b 0

:install_failed
echo.
echo Installation failed. Check your internet connection and Python installation.
echo The detailed error should be shown above.
pause
exit /b 1
