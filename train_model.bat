@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: Fine-tune Kraken OCR Model
:: Usage: train_model.bat <training_data_folder>

echo.
echo ============================================
echo   Kraken Model Fine-Tuning Tool
echo ============================================

if "%~1"=="" (
    echo.
    echo Usage: train_model.bat ^<training_data_folder^>
    echo.
    echo The training folder should contain:
    echo   - Image files: *.png, *.tif, *.jpg
    echo   - Ground truth: *.gt.txt [same name as image]
    echo.
    echo Example folder structure:
    echo   training_data/
    echo     line001.png
    echo     line001.gt.txt
    echo     line002.png
    echo     line002.gt.txt
    echo.
    echo Or use ALTO/PageXML format:
    echo   training_data/
    echo     page001.png
    echo     page001.xml
    echo.
    exit /b 1
)

set "TRAINING_DIR=%~1"
set "SCRIPT_DIR=%~dp0"
set "BASE_MODEL=%SCRIPT_DIR%models\arabPers-WithDiffTypefaces.mlmodel"
set "OUTPUT_PREFIX=%SCRIPT_DIR%models\fine_tuned"
set "PYTHON_EXE=C:\Users\AliTaghavi\AppData\Local\Programs\Python\Python311\python.exe"

:: Check if training directory exists
if not exist "%TRAINING_DIR%" (
    echo Error: Training directory not found: %TRAINING_DIR%
    exit /b 1
)

:: Check if base model exists
if not exist "%BASE_MODEL%" (
    echo Error: Base model not found: %BASE_MODEL%
    exit /b 1
)

echo.
echo Base Model: %BASE_MODEL%
echo Training Data: %TRAINING_DIR%
echo Output: %OUTPUT_PREFIX%
echo.
echo Starting training... [This may take several hours]
echo.

:: Set UTF-8 encoding
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

:: Run ketos train
ketos train ^
    -i "%BASE_MODEL%" ^
    -o "%OUTPUT_PREFIX%" ^
    --resize both ^
    -r 0.0001 ^
    --schedule cosine ^
    "%TRAINING_DIR%\*.png"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo   Training Complete!
    echo ============================================
    echo.
    echo Your fine-tuned model is saved in the models folder.
    echo To use it, update ocr.bat or run:
    echo   kraken -i image.png output.txt binarize segment ocr -m models/fine_tuned_best.mlmodel
) else (
    echo.
    echo Training encountered an error.
    echo Check the output above for details.
)

endlocal
