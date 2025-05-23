@echo off
REM Neural Noise Segmentation App - Quick Start
echo ============================================================
echo       Neural Noise Segmentation App - Quick Start
echo ============================================================
echo.

echo Checking Python installation...
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.7+ and try again.
    goto :end
)

cd /d "%~dp0"
echo Starting application with validation checks...
python quick_start.py

:end
echo.
echo Press any key to exit...
pause >nul
