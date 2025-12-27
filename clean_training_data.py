"""
Clean training data by removing files with empty or missing ground truth.
"""
import os
from pathlib import Path

DATA_DIR = Path("training_data_words")

def main():
    print(f"Cleaning training data in: {DATA_DIR}")

    png_files = list(DATA_DIR.glob("*.png"))
    print(f"Found {len(png_files)} PNG files")

    removed = 0
    valid = 0

    for png in png_files:
        gt = DATA_DIR / png.name.replace('.png', '.gt.txt')

        should_remove = False
        reason = ""

        if not gt.exists():
            should_remove = True
            reason = "GT missing"
        else:
            text = gt.read_text(encoding='utf-8').strip()
            if len(text) == 0:
                should_remove = True
                reason = "GT empty"

        if should_remove:
            # Remove both files
            png.unlink()
            if gt.exists():
                gt.unlink()
            removed += 1
            if removed <= 10:
                print(f"  Removed: {png.name} ({reason})")
        else:
            valid += 1

    if removed > 10:
        print(f"  ... and {removed - 10} more")

    print(f"\nResults:")
    print(f"  Removed: {removed} pairs")
    print(f"  Valid: {valid} pairs")
    print(f"\nReady for training!")

if __name__ == "__main__":
    main()
