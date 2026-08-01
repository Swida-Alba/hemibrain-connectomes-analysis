@echo off
REM =============================================================================
REM DROCAT UI Launcher for Windows
REM =============================================================================
REM Usage: Double-click or run: run_ui.bat
REM =============================================================================

echo Starting DROCAT UI...

REM Try to activate conda environment
call conda activate drocat 2>nul
if %ERRORLEVEL% NEQ 0 (
    REM Try common conda locations
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat" drocat
    ) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat" drocat
    ) else (
        echo Warning: Could not activate conda environment 'drocat'
        echo Trying to run with system Python...
    )
)

REM Launch UI
cd /d "%~dp0"
python ui/app.py
