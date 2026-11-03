@echo off
setlocal
set PYTHONUTF8=1
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B "%~dp0play.py" --model discrete %*
set "play_exit=%errorlevel%"
if not "%play_exit%"=="0" pause
exit /b %play_exit%
