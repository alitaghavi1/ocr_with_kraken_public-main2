import os
import sys

# Add CUDA to PATH
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
