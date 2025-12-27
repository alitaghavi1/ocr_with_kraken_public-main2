@echo off
REM Train Kraken from Scratch
REM =========================
REM
REM Train a completely new OCR model for your handwriting.
REM Best when you have lots of training data (1000+ samples) and
REM your handwriting is very different from existing models.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "C:\AR team\Ali\ocr_with_kraken_public-main"

echo ============================================================
echo Train Kraken OCR Model from Scratch
echo ============================================================
echo.
echo Training data: handwritten_training_data/
echo.
echo Note: Training from scratch requires significant data
echo       (1000+ line images recommended)
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
  "handwritten_training_data/*.png"

echo.
pause
