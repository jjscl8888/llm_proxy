@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_DIR=%SCRIPT_DIR%\.."

if "%~1"=="" (
    set "CONFIG=%PROJECT_DIR%\config.yaml"
) else (
    set "CONFIG=%~1"
)

cd /d "%PROJECT_DIR%"

start "LLM Proxy" uv run python -m llm_proxy --config "%CONFIG%"

timeout /t 2 /nobreak >nul

echo LLM Proxy started
echo.
echo To use with Claude Code CLI, run:
echo.
echo   set ANTHROPIC_BASE_URL=http://127.0.0.1:8082
echo   set ANTHROPIC_API_KEY=sk-proxy-pass-through
echo   claude
echo.

pause
