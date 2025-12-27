@echo off
REM Kraken OCR Fine-Tuning for Handwritten Data
REM ============================================
REM
REM Usage:
REM   run_training.bat              - Fine-tune from base model (or scratch if no base)
REM   run_training.bat --continue   - Continue from last checkpoint
REM   run_training.bat --scratch    - Train new model from scratch
REM
REM Before running:
REM   1. Place your training data in handwritten_training_data/
REM   2. Each image needs a .gt.txt file with the transcription
REM   3. (Optional) Set BASE_MODEL in train.py for fine-tuning

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "C:\Ali\ocr_with_kraken_public-main"

echo ============================================================
echo Kraken OCR Fine-Tuning for Handwritten Data
echo ============================================================
echo.

REM Check for training data
if not exist "handwritten_training_data\*.png" (
    echo ERROR: No training images found in handwritten_training_data/
    echo.
    echo Please prepare your training data first:
    echo   1. Put line images in handwritten_training_data/
    echo   2. Create .gt.txt file for each image with the transcription
    echo.
    echo Example:
    echo   handwritten_training_data/line001.png
    echo   handwritten_training_data/line001.gt.txt  ^<-- contains the text
    echo.
    pause
    exit /b 1
)

echo Training data found. Starting training...
echo Check training_log.txt for progress.
echo.

.venv\Scripts\python.exe run_training.py %*

echo.
pause
