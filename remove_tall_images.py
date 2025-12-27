"""
Remove images that are too tall (not single lines).
Single text lines should be max ~200-300px tall.
"""

from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"

MAX_HEIGHT = 300  # pixels - reasonable max for a single line

def main():
    print(f"Removing images taller than {MAX_HEIGHT}px")
    print("=" * 60)

    png_files = list(BALANCED_DIR.glob("*.png"))
    print(f"Total images: {len(png_files)}\n")

    removed = 0
    kept = 0

    for png_path in png_files:
        try:
            img = Image.open(png_path)
            w, h = img.size

            if h > MAX_HEIGHT:
                # Remove PNG and GT file
                gt_path = png_path.with_suffix('.gt.txt')
                png_path.unlink()
                if gt_path.exists():
                    gt_path.unlink()
                removed += 1
            else:
                kept += 1

        except Exception as e:
            print(f"  Error with {png_path.name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Removed: {removed}")
    print(f"  Kept: {kept}")
    print(f"  New total: {len(list(BALANCED_DIR.glob('*.png')))}")

if __name__ == "__main__":
    main()
