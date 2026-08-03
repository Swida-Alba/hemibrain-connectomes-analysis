@echo off
REM DROCAT one-click installer for Windows. The implementation lives in
REM install.ps1 so command-line and double-click installs cannot drift apart.
setlocal
set "SCRIPT_DIR=%~dp0"

where pwsh >nul 2>nul
if not errorlevel 1 (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install.ps1"
)
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" echo Installation failed with exit code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
