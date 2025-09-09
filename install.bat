@echo off
echo Installing Live Captions Computer Vision Translator...
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt

echo.
echo Installation complete!
echo.
echo To run the application:
echo   python Caption_live_translater.py
echo.
echo Make sure Tesseract OCR is installed at:
echo   C:\Program Files\Tesseract-OCR\tesseract.exe
echo.
pause
