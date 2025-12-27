@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Fine-tuning Kraken on Line-Level Data
echo  Base model: all_arabic_scripts.mlmodel
echo  Training data: ~48,636 line images
echo    - Muharaf: 24,495
echo    - KHATT: 6,673
echo    - BL RASM: 2,596
echo    - RASAM v3 (Maghrebi): 6,877
echo    - OpenITI Persian: 1,209
echo    - OpenITI Arabic: 1,843
echo    - Other: 2,612
echo  All images: grayscale
echo ============================================
echo.

echo Starting training...
.venv\Scripts\ketos.exe -d cuda:0 train ^
    -o models/line_finetuned ^
    -f path ^
    -B 16 ^
    -N 50 ^
    --lag 10 ^
    -r 0.0001 ^
    --schedule reduceonplateau ^
    -i models/all_arabic_scripts.mlmodel ^
    --resize new ^
    "training_data_lines/balanced_training/*.png"

echo.
echo Training complete!
pause
