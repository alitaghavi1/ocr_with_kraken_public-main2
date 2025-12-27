# Force remove images taller than 300px
Add-Type -AssemblyName System.Drawing

$folder = "C:\AR team\Ali\ocr_with_kraken_public-main\training_data_lines\balanced_training"
$maxHeight = 300
$removed = 0
$total = 0

Write-Host "Removing images taller than $maxHeight px" -ForegroundColor Cyan
Write-Host "Folder: $folder"
Write-Host "=========================================="

$files = Get-ChildItem -Path $folder -Filter "*.png"
$total = $files.Count
Write-Host "Total PNG files: $total"

foreach ($file in $files) {
    try {
        $stream = [System.IO.File]::OpenRead($file.FullName)
        $img = [System.Drawing.Image]::FromStream($stream, $false, $false)
        $height = $img.Height
        $img.Dispose()
        $stream.Close()
        $stream.Dispose()

        if ($height -gt $maxHeight) {
            # Force delete
            Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
            $gtFile = $file.FullName -replace '\.png$', '.gt.txt'
            Remove-Item -Path $gtFile -Force -ErrorAction SilentlyContinue
            $removed++

            if ($removed % 50 -eq 0) {
                Write-Host "  Removed: $removed"
            }
        }
    }
    catch {
        Write-Host "  Error: $($file.Name) - $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Done!" -ForegroundColor Green
Write-Host "  Removed: $removed"
Write-Host "  Remaining: $($total - $removed)"
