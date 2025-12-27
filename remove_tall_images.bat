@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Removing images taller than 300px from balanced_training
echo =========================================================
echo.

set COUNT=0
set FOLDER=training_data_lines\balanced_training

for %%f in ("%FOLDER%\*.png") do (
    powershell -Command "$img = [System.Drawing.Image]::FromFile('%%f'); $h = $img.Height; $img.Dispose(); if ($h -gt 300) { exit 1 } else { exit 0 }"
    if errorlevel 1 (
        del /f /q "%%f" 2>nul
        del /f /q "%%~dpnf.gt.txt" 2>nul
        set /a COUNT+=1
        echo Removed: %%~nxf
    )
)

echo.
echo =========================================================
echo Done! Removed %COUNT% tall images.
echo.
pause
