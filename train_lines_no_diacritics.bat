@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Fine-tuning Kraken - NO DIACRITICS
echo  Base model: all_arabic_scripts.mlmodel
echo  Training data: balanced_training_no_diacritics
echo  (Same images, ground truth without tashkeel)
echo ============================================
echo.

REM Check if stripped data exists
if not exist "training_data_lines\balanced_training_no_diacritics\*.png" (
    echo ERROR: Stripped training data not found!
    echo.
    echo Please run first:
    echo   .venv\Scripts\python.exe strip_diacritics.py
    echo.
    pause
    exit /b 1
)

echo Starting training without diacritics...
.venv\Scripts\ketos.exe -d cuda:0 train ^
    -o models/line_finetuned_no_diacritics ^
    -f path ^
    -B 8 ^
    -N 50 ^
    --lag 10 ^
    -r 0.0001 ^
    --schedule reduceonplateau ^
    --augment ^
    -i models/all_arabic_scripts.mlmodel ^
    --resize new ^
    "training_data_lines/balanced_training_no_diacritics/*.png"

echo.
echo Training complete!
pause
