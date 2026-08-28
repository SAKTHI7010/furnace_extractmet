@echo off
:: SmartMelt Studio — Web Application Launcher
:: Run this from the project root to start the Starlette web server.

title SmartMelt Studio Web

echo.
echo  ============================================================
echo   SmartMelt Studio  ^|  HTML + CSS + Three.js Web App
echo  ============================================================
echo.

:: Detect virtual environment
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PYTHON=venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo  Using Python: %PYTHON%
echo.

:: Check if uvicorn is available
%PYTHON% -c "import uvicorn" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo  Installing missing dependency: uvicorn ...
    %PYTHON% -m pip install uvicorn --quiet
)

:: Check if starlette is available
%PYTHON% -c "import starlette" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo  Installing missing dependency: starlette ...
    %PYTHON% -m pip install starlette --quiet
)

echo  Starting SmartMelt Studio on http://localhost:8765
echo  Press Ctrl+C to stop the server.
echo.

:: Open browser after brief delay (background)
start /b cmd /c "timeout /t 2 /nobreak >NUL && start http://localhost:8765"

:: Launch the Starlette ASGI server
%PYTHON% -m uvicorn web_server:app --host 0.0.0.0 --port 8765 --reload

:: If uvicorn fails, try direct
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [WARN] uvicorn failed, trying direct python launch...
    %PYTHON% web_server.py
)

pause
