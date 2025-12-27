@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  Fine-tuning Kraken on Line-Level Data
echo  Base model: all_arabic_scripts.mlmodel
echo  Training data: 24,495 line images
echo ============================================
echo.

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONLEGACYWINDOWSSTDIO=1

.venv\Scripts\python.exe -c "import os; os.environ['PYTHONIOENCODING']='utf-8'; os.environ['PYTHONUTF8']='1'; import subprocess; subprocess.run([r'.venv\Scripts\ketos.exe', '-d', 'cuda:0', 'train', '-o', 'models/line_finetuned', '-f', 'path', '-B', '8', '-N', '50', '--lag', '10', '-r', '0.0001', '--schedule', 'reduceonplateau', '--augment', '-i', 'models/all_arabic_scripts.mlmodel', '--resize', 'union', 'training_data_lines/public_line_images/*.png'], env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'})"

echo.
echo Training complete!
pause
