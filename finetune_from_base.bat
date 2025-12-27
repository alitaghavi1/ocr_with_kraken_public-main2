@echo off
REM Fine-tune Kraken from a downloaded base model
REM ==============================================
REM
REM This script fine-tunes a pre-trained Kraken model with your data.
REM
REM Before running:
REM   1. Download a base model: python download_base_model.py arabic_best
REM   2. Set the BASE_MODEL path below or in train.py
REM   3. Place training data in handwritten_training_data/

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "C:\AR team\Ali\ocr_with_kraken_public-main"

echo ============================================================
echo Fine-Tune Kraken from Base Model
echo ============================================================
echo.

REM Set your base model path here (or edit train.py)
set BASE_MODEL=models/arabic_best.mlmodel

if not exist "%BASE_MODEL%" (
    echo Base model not found: %BASE_MODEL%
    echo.
    echo Download a base model first:
    echo   .venv\Scripts\python.exe download_base_model.py --list
    echo   .venv\Scripts\python.exe download_base_model.py arabic_best
    echo.
    pause
    exit /b 1
)

echo Base model: %BASE_MODEL%
echo Training data: handwritten_training_data/
echo.

.venv\Scripts\ketos.exe -d cuda:0 train ^
  -i "%BASE_MODEL%" ^
  -o "models/fine_tuned" ^
  -f path ^
  --resize union ^
  -B 8 ^
  -N 50 ^
  --lag 10 ^
  -r 0.0001 ^
  --augment ^
  --schedule reduceonplateau ^
  -F 1 ^
  "handwritten_training_data/*.png"

echo.
pause
