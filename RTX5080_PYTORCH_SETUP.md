# PyTorch Setup for NVIDIA RTX 5080 (Blackwell Architecture)

This guide documents the steps to make PyTorch work with NVIDIA RTX 5080/5090 GPUs which use the Blackwell architecture (sm_120).

## The Problem

Standard PyTorch releases do not include CUDA kernels compiled for Blackwell GPUs. When you try to run PyTorch on an RTX 5080, you get:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

PyTorch can *detect* the GPU but cannot *run operations* on it because the CUDA kernels for sm_120 architecture are missing.

## Prerequisites

- Windows 11 (22H2 or later)
- NVIDIA Driver 570.00 or newer
- CUDA Toolkit 12.8+ (CUDA 13.0 also works)
- Python 3.10 or 3.11 (recommended)

## Solution: Install PyTorch Nightly with CUDA 12.8

PyTorch nightly builds include Blackwell (sm_120) support.

### Step 1: Create/Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or for Command Prompt
.venv\Scripts\activate.bat
```

### Step 2: Install PyTorch Nightly with CUDA 12.8

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This installs:
- `torch` (e.g., 2.11.0.dev+cu128)
- `torchvision` (e.g., 0.25.0.dev+cu128)
- `torchaudio` (e.g., 2.10.0.dev+cu128)

### Step 3: Verify Installation

Create a test script `test_gpu.py`:

```python
import os
import sys

# Add CUDA to PATH (adjust version if needed)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
os.environ["PATH"] = cuda_path + ";" + os.environ.get("PATH", "")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        t = torch.rand(3,3).cuda()
        print(f"GPU tensor test: PASSED - sum = {t.sum().item()}")
    else:
        print("CUDA not available")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

Run it:
```powershell
.\.venv\Scripts\python.exe test_gpu.py
```

Expected output:
```
PyTorch version: 2.11.0.dev20251226+cu128
CUDA available: True
CUDA version: 12.8
Device: NVIDIA GeForce RTX 5080
GPU tensor test: PASSED - sum = 4.421544075012207
```

### Step 4: Install Other Dependencies

After PyTorch is working, install your other dependencies:

```powershell
pip install -r requirements.txt
```

Note: You may see version conflict warnings (e.g., "requires torch<=2.9"). These are usually safe to ignore as long as the GPU test passes.

## Alternative: Custom PyTorch Build

If nightly builds don't work, there's a community project with custom Blackwell builds:

- Repository: https://github.com/kentstone84/pytorch-rtx5080-support
- Package: `pip install rtx-stone[all]`

However, the official PyTorch nightly is recommended as it's more actively maintained.

## Troubleshooting

### "CUDA not available" after installation

1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Check CUDA path is in environment: Add CUDA bin to PATH
3. Reinstall with `--force-reinstall` flag

### Version conflicts with other packages

Some packages (like `kraken`) may have strict version requirements. The nightly PyTorch usually works despite these warnings. If issues occur:

```powershell
# Downgrade numpy if needed
pip install "numpy>=2.0,<2.3"
```

### Performance optimization

Add this to your training scripts for better Tensor Core utilization:

```python
import torch
torch.set_float32_matmul_precision('medium')  # or 'high'
```

## Quick Reference Commands

```powershell
# Full installation sequence
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt

# Verify GPU
.\.venv\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

## References

- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
- [PyTorch Forums - sm_120 Support](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [RTX 5080 PyTorch Community Build](https://github.com/kentstone84/pytorch-rtx5080-support)

---
Last updated: December 2025
