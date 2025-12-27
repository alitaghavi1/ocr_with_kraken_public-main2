@echo off
REM Train Kraken with WORDS ONLY (no single characters)
REM ====================================================
REM
REM Single-character samples don't work well with Kraken.
REM This script trains using only multi-character (word) samples.
REM
REM First run: python filter_for_training.py
REM to create the training_data_words folder.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "C:\AR team\Ali\ocr_with_kraken_public-main"

echo ============================================================
echo Train Kraken OCR - Words Only (No Single Characters)
echo ============================================================
echo.

REM Check for filtered data
if not exist "training_data_words\*.png" (
    echo Training data not found. Creating filtered dataset...
    echo.
    .venv\Scripts\python.exe filter_for_training.py
    echo.
)

if not exist "training_data_words\*.png" (
    echo ERROR: No training data in training_data_words/
    echo Run: python filter_for_training.py
    pause
    exit /b 1
)

echo Starting training with word samples only...
echo.

.venv\Scripts\ketos.exe -d cuda:0 train ^
  -o "models/fine_tuned" ^
  -f path ^
  -B 8 ^
  -N 100 ^
  --lag 15 ^
  -r 0.001 ^
  --augment ^
  --schedule reduceonplateau ^
  -F 1 ^
  --spec "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,64 Do0.1,2 Mp2,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 O2s1c{chars}]" ^
  "training_data_words/*.png"

echo.
pause
