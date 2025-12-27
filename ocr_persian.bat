@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ============================================
REM  Persian OCR Script
REM  Change the IMAGE_PATH below to your image
REM ============================================

set IMAGE_PATH=examples\yamini_full.jpg
set IMAGE_PATH=examples\masnavi_8.png

set MODEL=models\all_arabic_scripts.mlmodel
set MODEL=models\line_finetuned_best.mlmodel

REM ============================================

echo.
echo Running OCR on: %IMAGE_PATH%
echo Model: %MODEL%
echo.

.venv\Scripts\python.exe ocr_image.py "%IMAGE_PATH%" --model "%MODEL%" --print

echo.
pause
