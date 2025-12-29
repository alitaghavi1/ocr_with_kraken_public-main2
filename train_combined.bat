@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Fine-tuning Kraken on Combined Data
echo  Base model: all_arabic_scripts.mlmodel
echo  Training data: ~155,000 images
echo    - Lines: 46,355 (full sentences)
echo    - Words: 108,685 (short sequences)
echo  All images: grayscale, 64px height
echo ============================================
echo.

echo Starting training...
.venv\Scripts\ketos.exe -d cuda:0 train ^
    -o models/combined_finetuned ^
    -f path ^
    -B 32 ^
    -N 50 ^
    --lag 10 ^
    -r 0.0001 ^
    --schedule reduceonplateau ^
    -i models/all_arabic_scripts.mlmodel ^
    --resize new ^
    "combined_training/*.png"

echo.
echo Training complete!
pause
