"""Check which file ranges have PNGs"""
from pathlib import Path

d = Path("handwritten_training_data")

# Find last PNG file number
pngs = sorted(d.glob("*.png"))
if pngs:
    print(f"PNG files: {len(pngs)}")
    print(f"First PNG: {pngs[0].name}")
    print(f"Last PNG: {pngs[-1].name}")

# Find last GT file number
gts = sorted(d.glob("*.gt.txt"))
if gts:
    print(f"\nGT files: {len(gts)}")
    print(f"First GT: {gts[0].name}")
    print(f"Last GT: {gts[-1].name}")

# Check content of last PNG's corresponding GT
if pngs:
    last_png = pngs[-1]
    last_gt = last_png.with_suffix(".gt.txt")
    if last_gt.exists():
        print(f"\nLast PNG ({last_png.name}) GT content: '{last_gt.read_text(encoding='utf-8').strip()}'")

# Check what's in GT files just after last PNG
    # Get the number from last png
    last_num = int(last_png.stem.split('_')[1])
    next_gt = d / f"hw_{last_num + 1:06d}.gt.txt"
    if next_gt.exists():
        print(f"Next GT ({next_gt.name}) content: '{next_gt.read_text(encoding='utf-8').strip()}'")
        next_png = next_gt.with_suffix(".png")
        print(f"Next PNG exists: {next_png.exists()}")
