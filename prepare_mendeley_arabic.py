"""
Prepare Mendeley Historical Arabic Dataset for Kraken training.

Source: https://data.mendeley.com/datasets/xz6f8bw3w8/1
License: CC BY 4.0

Instructions:
1. Download from Mendeley Data
2. Extract to training_data_lines/mendeley_arabic/
3. Run this script to convert to Kraken format
"""

import os
import shutil
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "training_data_lines" / "mendeley_arabic"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "mendeley_lines"


def find_image_text_pairs():
    """Find all image files and their corresponding text files."""
    pairs = []

    # Common image extensions
    img_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    # Walk through all directories
    for root, dirs, files in os.walk(INPUT_DIR):
        root_path = Path(root)

        for file in files:
            file_path = root_path / file

            # Check if it's an image
            if file_path.suffix.lower() in img_extensions:
                # Look for corresponding text file
                possible_gt = [
                    file_path.with_suffix('.txt'),
                    file_path.with_suffix('.gt.txt'),
                    root_path / (file_path.stem + '.txt'),
                    root_path / (file_path.stem + '_gt.txt'),
                ]

                gt_file = None
                for gt_path in possible_gt:
                    if gt_path.exists():
                        gt_file = gt_path
                        break

                if gt_file:
                    pairs.append((file_path, gt_file))

    return pairs


def convert_to_kraken_format(pairs):
    """Convert image-text pairs to Kraken training format."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for img_path, gt_path in pairs:
        try:
            # Read text
            text = gt_path.read_text(encoding='utf-8').strip()
            if not text:
                continue

            # Load and convert image
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')

            # Save with Kraken naming convention
            out_name = f"mendeley_{total_saved:05d}"
            out_png = OUTPUT_DIR / f"{out_name}.png"
            out_gt = OUTPUT_DIR / f"{out_name}.gt.txt"

            img.save(out_png)
            out_gt.write_text(text, encoding='utf-8')
            total_saved += 1

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    return total_saved


def explore_structure():
    """Explore the dataset structure to understand format."""
    print("Exploring dataset structure...")
    print()

    if not INPUT_DIR.exists():
        print(f"ERROR: Directory not found: {INPUT_DIR}")
        print()
        print("Please download the dataset first:")
        print("  1. Go to: https://data.mendeley.com/datasets/xz6f8bw3w8/1")
        print("  2. Click 'Download All'")
        print(f"  3. Extract to: {INPUT_DIR}")
        return False

    # List contents
    print(f"Contents of {INPUT_DIR}:")

    file_types = {}
    total_files = 0

    for root, dirs, files in os.walk(INPUT_DIR):
        rel_path = Path(root).relative_to(INPUT_DIR)
        if rel_path != Path('.'):
            print(f"\n  {rel_path}/")

        for file in files[:10]:  # Show first 10 files per directory
            suffix = Path(file).suffix.lower()
            file_types[suffix] = file_types.get(suffix, 0) + 1
            total_files += 1
            print(f"    {file}")

        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more files")

    print()
    print(f"Total files: {total_files}")
    print("File types:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count}")

    return True


def main():
    print("=" * 60)
    print("Mendeley Historical Arabic Dataset Processor")
    print("=" * 60)
    print()

    # Check if data exists
    if not explore_structure():
        return

    # Find pairs
    print()
    print("Looking for image-text pairs...")
    pairs = find_image_text_pairs()
    print(f"Found {len(pairs)} image-text pairs")

    if not pairs:
        print()
        print("No image-text pairs found.")
        print("The dataset structure may be different than expected.")
        print("Please check the downloaded files manually.")
        return

    # Convert
    print()
    print("Converting to Kraken format...")
    total = convert_to_kraken_format(pairs)

    print()
    print("=" * 60)
    print(f"Complete! Saved {total} training samples")
    print(f"Output: {OUTPUT_DIR}")
    print()
    print("Add to your training with:")
    print(f'  "{OUTPUT_DIR}/*.png"')


if __name__ == "__main__":
    main()
