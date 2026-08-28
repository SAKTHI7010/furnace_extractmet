@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Creating SmartMelt environment...
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -c "import streamlit,pyarrow; p=tuple(int(x) for x in streamlit.__version__.split('.')[:2]); assert p>=(1,60)" >nul 2>nul
if errorlevel 1 (
  echo Installing or upgrading SmartMelt dependencies...
  python -m pip install --upgrade pip
  python -m pip install --upgrade -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Dependency installation failed. Check the internet connection and try again.
    pause
    exit /b 1
  )
)
python -m streamlit run streamlit_app.py
endlocal
