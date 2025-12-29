@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ============================================
REM  Persian OCR Script with Post-Processing
REM  Change the IMAGE_PATH below to your image
REM ============================================

set IMAGE_PATH=examples\yamini_full.jpg

set MODEL=models\line_finetuned_52.mlmodel
set DICTIONARY=dictionaries\persian_dictionary_ganjoor.txt
set CONTEXT_MODEL=ocr_context_model.pkl

REM ============================================

REM Extract filename without extension for output
for %%F in ("%IMAGE_PATH%") do set BASENAME=%%~nF

set OCR_OUTPUT=OUTPUT\%BASENAME%_ocr.txt
set CORRECTED_OUTPUT=OUTPUT\%BASENAME%_corrected.txt

echo.
echo ============================================
echo  Persian OCR with Context-Aware Correction
echo ============================================
echo.
echo Image: %IMAGE_PATH%
echo Model: %MODEL%
echo.

REM Step 1: Run OCR
echo [Step 1] Running OCR...
.venv\Scripts\python.exe ocr_image.py "%IMAGE_PATH%" "%OCR_OUTPUT%" --model "%MODEL%"

if not exist "%OCR_OUTPUT%" (
    echo ERROR: OCR failed - no output created
    pause
    exit /b 1
)

REM Step 2: Run Post-Processing
echo.
echo [Step 2] Running post-processing with fuzzy matching...
.venv\Scripts\python.exe -c "import sys,io;sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace');from pathlib import Path;from post_process_context import ContextAwarePostProcessor;p=ContextAwarePostProcessor(dictionary_path=Path('dictionaries/persian_dictionary_ganjoor.txt'),fuzzy_threshold=80,context_weight=0.4);p.load_model(Path('ocr_context_model.pkl'));t=Path('%OCR_OUTPUT%').read_text(encoding='utf-8');c,corr=p.process_text(t,verbose=False);Path('%CORRECTED_OUTPUT%').write_text(c,encoding='utf-8');print(f'Words: {len(t.split())} | Corrections: {len(corr)}');[print(f'  {x[\"original\"]} -> {x[\"candidate\"]}') for x in corr[:10]]"

echo.
echo ============================================
echo  Results
echo ============================================
echo.
echo OCR Output:     %OCR_OUTPUT%
echo Corrected:      %CORRECTED_OUTPUT%
echo.
echo --- Corrected Text Preview ---
type "%CORRECTED_OUTPUT%"
echo.
echo ============================================
pause
