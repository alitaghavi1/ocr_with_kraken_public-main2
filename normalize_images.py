"""
Normalize all training images to grayscale (mode L) for consistent training.
"""

from pathlib import Path
from PIL import Image
from collections import Counter

BASE_DIR = Path(__file__).parent
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"

def main():
    png_files = list(BALANCED_DIR.glob("*.png"))
    print(f"Found {len(png_files)} images in balanced_training\n")

    # First, analyze current modes
    print("Analyzing image modes...")
    modes = Counter()
    for img_path in png_files:
        try:
            img = Image.open(img_path)
            modes[img.mode] += 1
        except:
            modes['ERROR'] += 1

    print("Current image modes:")
    for mode, count in modes.most_common():
        print(f"  {mode}: {count}")

    # Convert all to grayscale
    print(f"\nConverting all images to grayscale (L)...")
    converted = 0
    already_L = 0
    errors = 0

    for i, img_path in enumerate(png_files):
        try:
            img = Image.open(img_path)

            if img.mode != 'L':
                # Convert to grayscale
                img_gray = img.convert('L')
                img_gray.save(img_path)
                converted += 1
            else:
                already_L += 1

            if (i + 1) % 2000 == 0:
                print(f"  Processed {i + 1}/{len(png_files)}...")

        except Exception as e:
            print(f"  Error with {img_path.name}: {e}")
            errors += 1

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Already grayscale: {already_L}")
    print(f"  Converted to grayscale: {converted}")
    print(f"  Errors: {errors}")

if __name__ == "__main__":
    main()
