"""
Add OpenITI lines to balanced_training.
"""

from pathlib import Path
from PIL import Image
import shutil

BASE_DIR = Path(__file__).parent
OPENITI_DIR = BASE_DIR / "training_data_lines" / "openiti_lines"
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"


def main():
    print("Adding OpenITI lines to balanced_training")
    print("=" * 50)

    openiti_pngs = sorted(OPENITI_DIR.glob("*.png"))
    print(f"OpenITI lines to add: {len(openiti_pngs)}")

    existing = list(BALANCED_DIR.glob("*.png"))
    print(f"Existing balanced_training: {len(existing)}")

    # Remove any existing openiti_* files
    existing_openiti = list(BALANCED_DIR.glob("openiti_*.png"))
    if existing_openiti:
        print(f"Removing {len(existing_openiti)} existing openiti_* files")
        for f in existing_openiti:
            f.unlink()
            gt = f.with_suffix('.gt.txt')
            if gt.exists():
                gt.unlink()

    copied = 0

    for png_path in openiti_pngs:
        gt_path = png_path.with_suffix('.gt.txt')

        if not gt_path.exists():
            continue

        new_png = BALANCED_DIR / f"openiti_{copied:05d}.png"
        new_gt = BALANCED_DIR / f"openiti_{copied:05d}.gt.txt"

        try:
            img = Image.open(png_path)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(new_png)

            shutil.copy2(gt_path, new_gt)
            copied += 1

            if copied % 500 == 0:
                print(f"  Copied: {copied}")

        except Exception as e:
            print(f"  Error: {png_path.name} - {e}")

    print(f"\n{'=' * 50}")
    print(f"Done!")
    print(f"  Copied: {copied}")
    print(f"  Total in balanced_training: {len(list(BALANCED_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
