"""
Check training images for potential issues:
- Very large images (can cause memory issues)
- Corrupted images
- Unusual dimensions
"""

from pathlib import Path
from PIL import Image
import os

BASE_DIR = Path(__file__).parent
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"

def main():
    print("Checking training images for issues...")
    print("=" * 60)

    png_files = list(BALANCED_DIR.glob("*.png"))
    print(f"Total images: {len(png_files)}\n")

    large_images = []
    corrupted = []
    very_wide = []
    very_tall = []

    for i, png_path in enumerate(png_files):
        if (i + 1) % 5000 == 0:
            print(f"  Checked {i+1}/{len(png_files)}...")

        try:
            img = Image.open(png_path)
            w, h = img.size
            pixels = w * h

            # Check for very large images (>5 million pixels)
            if pixels > 5_000_000:
                large_images.append((png_path.name, w, h, pixels))

            # Check for unusual dimensions
            if w > 5000:
                very_wide.append((png_path.name, w, h))
            if h > 500:
                very_tall.append((png_path.name, w, h))

            # Try to load image data to check for corruption
            img.load()

        except Exception as e:
            corrupted.append((png_path.name, str(e)))

    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Total checked: {len(png_files)}")
    print(f"  Corrupted: {len(corrupted)}")
    print(f"  Very large (>5MP): {len(large_images)}")
    print(f"  Very wide (>5000px): {len(very_wide)}")
    print(f"  Very tall (>500px): {len(very_tall)}")

    if corrupted:
        print(f"\nCorrupted images:")
        for name, err in corrupted[:10]:
            print(f"  {name}: {err}")

    if large_images:
        print(f"\nLarge images (top 10):")
        large_images.sort(key=lambda x: x[3], reverse=True)
        for name, w, h, px in large_images[:10]:
            print(f"  {name}: {w}x{h} = {px:,} pixels")

    if very_tall:
        print(f"\nVery tall images (top 10):")
        very_tall.sort(key=lambda x: x[2], reverse=True)
        for name, w, h in very_tall[:10]:
            print(f"  {name}: {w}x{h}")

if __name__ == "__main__":
    main()
