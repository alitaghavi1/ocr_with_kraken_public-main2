# PyTorch RTX 5080 Installer for Windows 11
# Run this inside your activated virtual environment

Write-Host "`nInstalling PyTorch with RTX 5080 support..." -ForegroundColor Cyan

# ---- Check Python version ----------------------------------------------------
$pyVersion = (& python --version) 2>$null
Write-Host "Detected $pyVersion"

# Get exit code from inline Python check
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Python 3.10 or higher required. Current: $pyVersion"
    exit 1
}

# ---- Install required dependencies ------------------------------------------
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install --quiet filelock fsspec Jinja2 MarkupSafe mpmath networkx sympy "typing_extensions>=4.10.0"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to install dependencies."
    exit 1
}
# ---- Locate site-packages and copy torch ------------------------------------
$sitePackages = python -c "import site; print(site.getsitepackages()[0])"
if (-not (Test-Path $sitePackages)) {
    Write-Error "Could not locate site-packages directory."
    exit 1
}

Write-Host "`nInstalling to: $sitePackages" -ForegroundColor Yellow

# Use PSScriptRoot to get the directory containing this script
$torchSource = Join-Path $PSScriptRoot "torch"

Write-Host "Looking for torch at: $torchSource" -ForegroundColor Yellow

if (-not (Test-Path $torchSource)) {
    Write-Error "❌ torch folder not found at: $torchSource"
    exit 1
}

Write-Host "Copying torch folder (this may take a minute)..." -ForegroundColor Yellow

try {
    # Remove old torch if it exists
    $torchDest = Join-Path $sitePackages "torch"
    if (Test-Path $torchDest) {
        Write-Host "Removing existing torch installation..." -ForegroundColor Yellow
        Remove-Item $torchDest -Recurse -Force -ErrorAction Stop
    }
    
    Copy-Item $torchSource -Destination $sitePackages -Recurse -Force -ErrorAction Stop
    Write-Host "✓ Copy complete" -ForegroundColor Green
}
catch {
    Write-Error "❌ Failed to copy torch: $_"
    exit 1
}

# ---- Verify installation ----------------------------------------------------
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:            {torch.cuda.get_device_name(0)}')
    print(f'Arch list:      {torch.cuda.get_arch_list()}')
"@

Write-Host "`n✅ Installation complete!" -ForegroundColor Green