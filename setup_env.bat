@echo off
REM Setup script for OCR with Kraken on a new Windows machine with GPU
REM Run this from the ocr_with_kraken_public-main directory

echo === Setting up Kraken OCR environment with GPU support ===

REM Check for Python 3.11
python --version 2>nul | findstr "3.11" >nul
if errorlevel 1 (
    echo Python 3.11 not found as default. Please ensure Python 3.11 is installed.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Remove old venv if it exists
if exist .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

REM Create new venv
echo Creating virtual environment...
python -m venv .venv

REM Activate venv
call .venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (CUDA 11.8)
echo Installing PyTorch with CUDA 11.8 support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

REM Install Kraken and dependencies
echo Installing Kraken OCR...
pip install kraken

REM Verify installation
echo.
echo === Verifying installation ===
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo === Setup complete! ===
echo.
echo To continue training, run: continue_training.bat
echo.
pause
