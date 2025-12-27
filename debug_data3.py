"""Check specific files"""
from pathlib import Path
import os

d = Path("handwritten_training_data")

# Check specific file that was reported as missing
test_files = ["hw_004657", "hw_004658", "hw_100000", "hw_200000"]

for name in test_files:
    gt = d / f"{name}.gt.txt"
    png = d / f"{name}.png"

    print(f"\n{name}:")
    print(f"  GT exists: {gt.exists()}")
    print(f"  PNG exists: {png.exists()}")

    if gt.exists():
        content = gt.read_text(encoding='utf-8').strip()
        print(f"  GT content: '{content}' (len={len(content)})")

    # Also check with os.path
    gt_os = os.path.join("handwritten_training_data", f"{name}.gt.txt")
    png_os = os.path.join("handwritten_training_data", f"{name}.png")
    print(f"  os.path GT exists: {os.path.exists(gt_os)}")
    print(f"  os.path PNG exists: {os.path.exists(png_os)}")

# List actual files in that range
print("\n\nActual files in hw_00465* range:")
for f in sorted(d.glob("hw_00465*")):
    print(f"  {f.name}")
