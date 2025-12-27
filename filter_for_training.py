"""
Filter training data to keep ONLY multi-character samples.
Single-character samples don't work well with Kraken's CTC loss.

This creates a new folder with only word/line samples for training.
"""

import os
import shutil
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "handwritten_training_data"
OUTPUT_DIR = Path(__file__).parent / "training_data_words"
MIN_CHARS = 2  # Minimum characters per sample


def main():
    print("Filtering training data for multi-character samples only...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Minimum characters: {MIN_CHARS}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_files = list(SOURCE_DIR.glob("*.gt.txt"))
    total = len(gt_files)
    print(f"Found {total} ground truth files")

    copied = 0
    skipped_single = 0
    skipped_empty = 0
    char_counts = {}

    for i, gt_path in enumerate(gt_files):
        if i % 50000 == 0 and i > 0:
            print(f"  Processing: {i}/{total} (copied: {copied})")

        try:
            text = gt_path.read_text(encoding='utf-8').strip()
            char_count = len(text)

            if char_count == 0:
                skipped_empty += 1
                continue

            if char_count < MIN_CHARS:
                skipped_single += 1
                continue

            # Track character distribution
            char_counts[char_count] = char_counts.get(char_count, 0) + 1

            # Copy files - handle .gt.txt -> .png correctly
            # gt_path.stem gives "hw_000000.gt", we need "hw_000000.png"
            img_name = gt_path.name.replace('.gt.txt', '.png')
            img_path = gt_path.parent / img_name
            if img_path.exists():
                shutil.copy(gt_path, OUTPUT_DIR / gt_path.name)
                shutil.copy(img_path, OUTPUT_DIR / img_path.name)
                copied += 1

        except Exception as e:
            print(f"  Error: {gt_path.name}: {e}")

    print()
    print("=" * 50)
    print(f"Results:")
    print(f"  Total files scanned: {total}")
    print(f"  Copied (multi-char): {copied}")
    print(f"  Skipped (single-char): {skipped_single}")
    print(f"  Skipped (empty): {skipped_empty}")
    print()
    print("Character count distribution:")
    for chars, count in sorted(char_counts.items())[:20]:
        print(f"  {chars} chars: {count} samples")
    if len(char_counts) > 20:
        print(f"  ... and {len(char_counts) - 20} more")
    print()
    print(f"Training data ready in: {OUTPUT_DIR}")
    print()
    print("To train, run:")
    print('  .venv\\Scripts\\ketos.exe -d cuda:0 train -o "models/fine_tuned" -f path -B 8 -N 100 --lag 15 -r 0.001 --augment --schedule reduceonplateau -F 1 --spec "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,64 Do0.1,2 Mp2,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 O2s1c{chars}]" "training_data_words/*.png"')


if __name__ == "__main__":
    main()
