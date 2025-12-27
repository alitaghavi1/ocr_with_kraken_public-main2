"""
Filter training data to reduce samples.
Limits single-char to 10k max, multi-char to 20k max.
"""

import os
import random
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).parent / "handwritten_training_data"
BACKUP_DIR = Path(__file__).parent / "handwritten_training_data_backup"
MAX_SINGLE_CHAR = 10000
MAX_MULTI_CHAR = 20000

def move_files_to_backup(files_to_remove, description):
    """Move file pairs to backup directory"""
    if not files_to_remove:
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nMoving {len(files_to_remove)} {description} pairs to backup...")

    for i, gt_path in enumerate(files_to_remove):
        if i % 5000 == 0:
            print(f"  Moving: {i}/{len(files_to_remove)}")

        # Move .gt.txt file
        shutil.move(str(gt_path), str(BACKUP_DIR / gt_path.name))

        # Move corresponding image file
        img_path = gt_path.with_suffix('.png')
        if img_path.exists():
            shutil.move(str(img_path), str(BACKUP_DIR / img_path.name))

def main():
    print("Scanning training data...")

    single_char_files = []
    multi_char_files = []

    gt_files = list(TRAINING_DIR.glob("*.gt.txt"))
    total = len(gt_files)

    print(f"Found {total} ground truth files")

    for i, gt_path in enumerate(gt_files):
        if i % 10000 == 0:
            print(f"  Scanning: {i}/{total}")

        try:
            text = gt_path.read_text(encoding='utf-8').strip()
            char_count = len(text)

            if char_count == 1:
                single_char_files.append(gt_path)
            elif char_count > 1:
                multi_char_files.append(gt_path)
        except Exception as e:
            print(f"  Error reading {gt_path.name}: {e}")

    print(f"\nResults:")
    print(f"  Single character samples: {len(single_char_files)}")
    print(f"  Multi-character samples: {len(multi_char_files)}")

    random.seed(42)  # Reproducible selection

    # Filter single-char files
    if len(single_char_files) > MAX_SINGLE_CHAR:
        random.shuffle(single_char_files)
        files_to_remove = single_char_files[MAX_SINGLE_CHAR:]
        print(f"\nWill keep {MAX_SINGLE_CHAR} single-char, remove {len(files_to_remove)}")
        move_files_to_backup(files_to_remove, "single-char")
        single_char_files = single_char_files[:MAX_SINGLE_CHAR]

    # Filter multi-char files
    if len(multi_char_files) > MAX_MULTI_CHAR:
        random.shuffle(multi_char_files)
        files_to_remove = multi_char_files[MAX_MULTI_CHAR:]
        print(f"\nWill keep {MAX_MULTI_CHAR} multi-char, remove {len(files_to_remove)}")
        move_files_to_backup(files_to_remove, "multi-char")
        multi_char_files = multi_char_files[:MAX_MULTI_CHAR]

    # Final count
    remaining_gt = len(list(TRAINING_DIR.glob("*.gt.txt")))
    print(f"\nFinal training set: {remaining_gt} samples")
    print(f"  - Multi-char: {len(multi_char_files)}")
    print(f"  - Single-char: {len(single_char_files)}")

if __name__ == "__main__":
    main()
