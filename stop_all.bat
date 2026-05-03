@echo off
title Vox Stack — Stop all
echo Stopping Vox stack services...

for %%P in (8000 8765 5173) do (
  for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%%P " ^| findstr "LISTENING"') do (
    echo   - killing PID %%a on port %%P
    taskkill /f /pid %%a >nul 2>&1
  )
)

taskkill /f /im ngrok.exe >nul 2>&1
echo Done.
pause
