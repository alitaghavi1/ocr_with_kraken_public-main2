@echo off
REM Continue Training from Last Checkpoint
REM =======================================
REM
REM This script continues training from the best model checkpoint.
REM Use this when you want to add more training data or continue
REM a training session that was interrupted.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "C:\AR team\Ali\ocr_with_kraken_public-main"

echo ============================================================
echo Continue Kraken Training from Checkpoint
echo ============================================================
echo.

REM Check for checkpoint model
if exist "models\fine_tuned_best.mlmodel" (
    set CHECKPOINT=models/fine_tuned_best.mlmodel
) else if exist "models\fine_tuned_17.mlmodel" (
    set CHECKPOINT=models/fine_tuned_17.mlmodel
) else (
    echo ERROR: No checkpoint model found!
    echo.
    echo Expected one of:
    echo   - models/fine_tuned_best.mlmodel
    echo   - models/fine_tuned_17.mlmodel
    echo.
    echo Run initial training first with: run_training.bat --scratch
    pause
    exit /b 1
)

echo Checkpoint: %CHECKPOINT%
echo Training data: handwritten_training_data/
echo.

.venv\Scripts\ketos.exe -d cuda:0 train ^
  -i "%CHECKPOINT%" ^
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
