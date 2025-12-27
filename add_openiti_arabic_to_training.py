"""
Add Arabic OpenITI lines to balanced_training.
"""

from pathlib import Path
from PIL import Image
import shutil

BASE_DIR = Path(__file__).parent
OPENITI_AR_DIR = BASE_DIR / "training_data_lines" / "openiti_lines_arabic"
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"


def main():
    print("Adding Arabic OpenITI lines to balanced_training")
    print("=" * 50)

    pngs = sorted(OPENITI_AR_DIR.glob("*.png"))
    print(f"Arabic OpenITI lines to add: {len(pngs)}")

    existing = list(BALANCED_DIR.glob("*.png"))
    print(f"Existing balanced_training: {len(existing)}")

    # Remove any existing openiti_ar_* files
    existing_ar = list(BALANCED_DIR.glob("openiti_ar_*.png"))
    if existing_ar:
        print(f"Removing {len(existing_ar)} existing openiti_ar_* files")
        for f in existing_ar:
            f.unlink()
            gt = f.with_suffix('.gt.txt')
            if gt.exists():
                gt.unlink()

    copied = 0

    for png_path in pngs:
        gt_path = png_path.with_suffix('.gt.txt')

        if not gt_path.exists():
            continue

        new_png = BALANCED_DIR / f"openiti_ar_{copied:05d}.png"
        new_gt = BALANCED_DIR / f"openiti_ar_{copied:05d}.gt.txt"

        try:
            img = Image.open(png_path)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(new_png)

            shutil.copy2(gt_path, new_gt)
            copied += 1

        except Exception as e:
            pass

    print(f"\nDone! Copied: {copied}")
    print(f"Total in balanced_training: {len(list(BALANCED_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
