"""Debug script to check training data"""
from pathlib import Path

d = Path("handwritten_training_data")

# Check first few GT files with 2+ chars
print("Checking multi-char samples...")
count = 0
found_with_png = 0
found_without_png = 0

for gt in d.glob("*.gt.txt"):
    text = gt.read_text(encoding="utf-8").strip()
    if len(text) >= 2:
        png = gt.with_suffix(".png")
        if png.exists():
            found_with_png += 1
            if count < 5:
                print(f"  {gt.name}: '{text}' -> PNG exists: YES")
        else:
            found_without_png += 1
            if count < 5:
                print(f"  {gt.name}: '{text}' -> PNG exists: NO")
        count += 1
        if count >= 20:
            break

print(f"\nOut of first {count} multi-char samples:")
print(f"  With PNG: {found_with_png}")
print(f"  Without PNG: {found_without_png}")

# Also check file naming
print("\nSample file names:")
gts = list(d.glob("*.gt.txt"))[:3]
pngs = list(d.glob("*.png"))[:3]
print(f"GT files: {[g.name for g in gts]}")
print(f"PNG files: {[p.name for p in pngs]}")
