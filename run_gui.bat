@echo off
echo Starting Medical Waste Feature Extractor GUI...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.6+ and try again
    pause
    exit /b 1
)

REM Check if PyQt5 is installed
python -c "import PyQt5" >nul 2>&1
if errorlevel 1 (
    echo PyQt5 is not installed. Installing now...
    pip install PyQt5
    if errorlevel 1 (
        echo Error: Failed to install PyQt5
        echo Please install PyQt5 manually: pip install PyQt5
        pause
        exit /b 1
    )
)

REM Check if other required packages are installed
python -c "import cv2, numpy, pandas, matplotlib, sklearn, joblib" >nul 2>&1
if errorlevel 1 (
    echo Some required packages are missing. Installing now...
    pip install opencv-python numpy pandas matplotlib scikit-learn joblib scikit-image
    if errorlevel 1 (
        echo Warning: Some packages failed to install
        echo The application may not work properly
    )
)

echo All dependencies checked. Starting GUI...
echo.

REM Run the GUI
python medical_waste_gui.py

if errorlevel 1 (
    echo.
    echo Error: GUI failed to start
    echo Please check the error messages above
    pause
)
