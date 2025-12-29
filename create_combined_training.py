"""
Create a combined training dataset from lines and words.

This merges:
- training_data_lines/balanced_training/ (46K line images)
- training_data_words/ (108K word images)

Into a single folder for training.
"""

import shutil
from pathlib import Path
from PIL import Image
import sys

# Paths
BASE_DIR = Path(__file__).parent
LINE_DIR = BASE_DIR / "training_data_lines" / "balanced_training"
WORD_DIR = BASE_DIR / "training_data_words"
OUTPUT_DIR = BASE_DIR / "combined_training"

# Settings
TARGET_HEIGHT = 64  # Standardize all images to this height


def get_valid_pairs(folder, pattern="*.png"):
    """Get all image files that have corresponding .gt.txt with content."""
    pairs = []
    for png_path in sorted(folder.glob(pattern)):
        gt_path = png_path.with_suffix('.gt.txt')
        if gt_path.exists():
            try:
                text = gt_path.read_text(encoding='utf-8').strip()
                if text:  # Non-empty ground truth
                    pairs.append((png_path, gt_path, text))
            except:
                pass
    return pairs


def resize_to_height(img, target_height):
    """Resize image to target height, maintaining aspect ratio."""
    if img.height == target_height:
        return img
    if img.height == 0:
        return img
    ratio = target_height / img.height
    new_width = max(1, int(img.width * ratio))
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def copy_with_resize(src_png, dest_png, target_height):
    """Copy image, resizing to target height and converting to grayscale."""
    img = Image.open(src_png)

    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Resize if needed
    if img.height != target_height:
        img = resize_to_height(img, target_height)

    img.save(dest_png)


def main():
    print("=" * 60)
    print("Creating Combined Training Dataset")
    print("=" * 60)

    # Check source directories
    if not LINE_DIR.exists():
        print(f"ERROR: Line directory not found: {LINE_DIR}")
        sys.exit(1)
    if not WORD_DIR.exists():
        print(f"ERROR: Word directory not found: {WORD_DIR}")
        sys.exit(1)

    # Get valid pairs from each source
    print(f"\nScanning line data: {LINE_DIR}")
    line_pairs = get_valid_pairs(LINE_DIR)
    print(f"  Found {len(line_pairs)} valid line images")

    print(f"\nScanning word data: {WORD_DIR}")
    word_pairs = get_valid_pairs(WORD_DIR)
    print(f"  Found {len(word_pairs)} valid word images")

    total = len(line_pairs) + len(word_pairs)
    print(f"\nTotal combined: {total} images")

    # Check if output exists
    if OUTPUT_DIR.exists():
        existing = len(list(OUTPUT_DIR.glob("*.png")))
        if existing > 0:
            print(f"\nOutput directory already has {existing} images.")
            response = input("Delete and recreate? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
            shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy files with unified naming
    print(f"\nCopying to: {OUTPUT_DIR}")
    print(f"Target height: {TARGET_HEIGHT}px")

    idx = 0

    # Copy line images first (they're the primary data)
    print("\nCopying line images...")
    for i, (png_path, gt_path, text) in enumerate(line_pairs):
        dest_png = OUTPUT_DIR / f"combined_{idx:06d}.png"
        dest_gt = OUTPUT_DIR / f"combined_{idx:06d}.gt.txt"

        copy_with_resize(png_path, dest_png, TARGET_HEIGHT)
        dest_gt.write_text(text, encoding='utf-8')

        idx += 1
        if (i + 1) % 5000 == 0:
            print(f"  Lines: {i + 1}/{len(line_pairs)}")

    print(f"  Lines complete: {len(line_pairs)} copied")

    # Copy word images
    print("\nCopying word images...")
    for j, (png_path, gt_path, text) in enumerate(word_pairs):
        dest_png = OUTPUT_DIR / f"combined_{idx:06d}.png"
        dest_gt = OUTPUT_DIR / f"combined_{idx:06d}.gt.txt"

        copy_with_resize(png_path, dest_png, TARGET_HEIGHT)
        dest_gt.write_text(text, encoding='utf-8')

        idx += 1
        if (j + 1) % 10000 == 0:
            print(f"  Words: {j + 1}/{len(word_pairs)}")

    print(f"  Words complete: {len(word_pairs)} copied")

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images: {idx}")
    print(f"  - From lines: {len(line_pairs)}")
    print(f"  - From words: {len(word_pairs)}")
    print(f"\nTo train with this data, use:")
    print(f'  "combined_training/*.png"')


if __name__ == "__main__":
    main()
