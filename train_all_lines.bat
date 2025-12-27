@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Fine-tuning Kraken on ALL Line-Level Data
echo  Base model: all_arabic_scripts.mlmodel
echo ============================================
echo.
echo Datasets:
echo   - public_line_images: ~24,495 lines (handwritten)
echo   - Arshasb_extracted: ~140,000 lines (typed Persian)
echo.

.venv\Scripts\ketos.exe -d cuda:0 train ^
    -o models/combined_finetuned ^
    -f path ^
    -B 8 ^
    -N 50 ^
    --lag 10 ^
    -r 0.0001 ^
    --schedule reduceonplateau ^
    --augment ^
    -i models/all_arabic_scripts.mlmodel ^
    --resize union ^
    "training_data_lines/public_line_images/*.png" ^
    "training_data_lines/Arshasb_extracted/*.png"

echo.
echo Training complete!
echo Model saved to: models/combined_finetuned_best.mlmodel
pause
