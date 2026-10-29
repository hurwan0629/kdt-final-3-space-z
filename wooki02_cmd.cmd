@echo off
chcp 65001 >nul
title WOOKI02 Training Monitor
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"%~dp0..\.venv-rl\Scripts\python.exe" -u "%~dp0scripts\watch_wooki02_cmd.py"
pause
