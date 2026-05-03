@echo off
title Vox Stack Launcher (LiveKit edition)
setlocal enabledelayedexpansion

set LAB_DIR=C:\Users\Praneeth p\OneDrive\Desktop\nexuscaller-lab
set NEXUS_DIR=C:\Users\Praneeth p\OneDrive\Desktop\NexusAgent
set FRONT_DIR=%NEXUS_DIR%\frontend

echo.
echo  ========================================
echo   Vox stack — LiveKit edition
echo   Starting 5 services in separate windows
echo  ========================================
echo.

:: ── 1. Free up the ports (kill stale processes from prior runs) ─────────
echo [1/6] Freeing ports 8000, 8765, 5173 + any stale ngrok tunnels...
call :kill_port 8000
call :kill_port 8765
call :kill_port 5173
taskkill /f /im ngrok.exe >nul 2>&1
timeout /t 2 /nobreak >nul

:: ── 2. LiveKit Agent worker (Vox brain — STT/LLM/TTS pipeline) ──────────
echo [2/6] Starting Vox Agent worker  (registers with LiveKit Cloud)...
start "Lab · Vox Agent worker" cmd /k ^
  "cd /d "%LAB_DIR%" && call venv\Scripts\activate && python -m voice_agent.agent dev"
timeout /t 4 /nobreak >nul

:: ── 3. Voice-agent FastAPI server (precall page + dial endpoint + cockpit) ─
echo [3/6] Starting Voice-agent server  (port 8765)...
start "Lab · Voice agent (8765)" cmd /k ^
  "cd /d "%LAB_DIR%" && call venv\Scripts\activate && uvicorn voice_agent.server:app --host 0.0.0.0 --port 8765 --reload"
timeout /t 3 /nobreak >nul

:: ── 4. NexusAgent API (CRM backend) ─────────────────────────────────────
echo [4/6] Starting NEXUSAGENT api  (port 8000)...
start "NexusAgent · API (8000)" cmd /k ^
  "cd /d "%NEXUS_DIR%" && call venv\Scripts\activate && uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

:: ── 5. NexusAgent frontend (Vite dev) ───────────────────────────────────
echo [5/6] Starting NEXUSAGENT frontend  (port 5173)...
start "NexusAgent · Frontend (5173)" cmd /k ^
  "cd /d "%FRONT_DIR%" && npm run dev -- --port 5173"
timeout /t 3 /nobreak >nul

:: ── 6. ngrok — exposes the voice-agent server publicly ──────────────────
::    With LiveKit, ngrok is OPTIONAL during dev: Twilio doesn't need to
::    reach this server anymore (LiveKit handles audio via SIP). ngrok is
::    only useful if you want to share the precall/cockpit page with someone
::    not on your local network. Keep it for parity with the old setup.
echo [6/6] Starting ngrok  (tunnels port 8765 → public; optional)...
start "ngrok · tunnel (lab → public)" cmd /k ^
  "ngrok http 8765"
timeout /t 2 /nobreak >nul

echo.
echo  ========================================
echo   All services launching in 5 windows.
echo.
echo   Vox Agent worker  : check that window for "registered worker"
echo   Lab API           : http://localhost:8765/health
echo   NexusAgent API    : http://localhost:8000/docs
echo   NexusAgent UI     : http://localhost:5173
echo   ngrok             : http://localhost:4040 (URL inside ngrok window)
echo.
echo   ⚠ The agent worker needs LIVEKIT_URL/KEY/SECRET in .env to register.
echo     If it fails, see voice_agent\README.md for the LiveKit setup.
echo  ========================================
echo.
echo Press any key to STOP everything (closes all windows)...
pause >nul

echo.
echo Stopping services...
taskkill /fi "WINDOWTITLE eq Lab · Vox Agent worker*"     /f /t >nul 2>&1
taskkill /fi "WINDOWTITLE eq Lab · Voice agent (8765)*"   /f /t >nul 2>&1
taskkill /fi "WINDOWTITLE eq NexusAgent · API (8000)*"    /f /t >nul 2>&1
taskkill /fi "WINDOWTITLE eq NexusAgent · Frontend (5173)*" /f /t >nul 2>&1
taskkill /fi "WINDOWTITLE eq ngrok · tunnel (lab → public)*" /f /t >nul 2>&1
call :kill_port 8000
call :kill_port 8765
call :kill_port 5173
taskkill /f /im ngrok.exe >nul 2>&1
echo Done.
exit /b 0


:kill_port
set PORT=%~1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
  echo   - killing PID %%a on port %PORT%
  taskkill /f /pid %%a >nul 2>&1
)
exit /b 0
