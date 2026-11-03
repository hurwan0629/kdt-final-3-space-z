@echo off
setlocal
set PYTHONUTF8=1
pushd "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Open play_discrete.bat or play_rotation.bat.
popd
pause
exit /b 0
:failed
echo Setup failed. Install 64-bit Python 3.12 and check the message above.
popd
pause
exit /b 1
