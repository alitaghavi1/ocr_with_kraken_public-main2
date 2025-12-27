"""
Add properly extracted RASAM v3 lines to balanced_training.
"""

from pathlib import Path
from PIL import Image
import shutil

BASE_DIR = Path(__file__).parent
RASAM_DIR = BASE_DIR / "training_data_lines" / "rasam_lines_v3"
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"


def main():
    print("Adding RASAM v3 lines to balanced_training")
    print("=" * 50)

    rasam_pngs = sorted(RASAM_DIR.glob("*.png"))
    print(f"RASAM v3 lines to add: {len(rasam_pngs)}")

    existing = list(BALANCED_DIR.glob("*.png"))
    print(f"Existing balanced_training: {len(existing)}")

    # Check if any rasam3_* files already exist
    existing_rasam3 = list(BALANCED_DIR.glob("rasam3_*.png"))
    if existing_rasam3:
        print(f"Found {len(existing_rasam3)} existing rasam3_* files - removing them first")
        for f in existing_rasam3:
            f.unlink()
            gt = f.with_suffix('.gt.txt')
            if gt.exists():
                gt.unlink()

    copied = 0
    skipped = 0

    for png_path in rasam_pngs:
        gt_path = png_path.with_suffix('.gt.txt')

        if not gt_path.exists():
            skipped += 1
            continue

        # Use rasam3_ prefix to distinguish from any old rasam_ files
        new_png = BALANCED_DIR / f"rasam3_{copied:05d}.png"
        new_gt = BALANCED_DIR / f"rasam3_{copied:05d}.gt.txt"

        try:
            # Convert to grayscale if needed
            img = Image.open(png_path)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(new_png)

            shutil.copy2(gt_path, new_gt)
            copied += 1

            if copied % 1000 == 0:
                print(f"  Copied: {copied}")

        except Exception as e:
            print(f"  Error: {png_path.name} - {e}")
            skipped += 1

    print(f"\n{'=' * 50}")
    print(f"Done!")
    print(f"  Copied: {copied}")
    print(f"  Skipped: {skipped}")
    print(f"  Total in balanced_training: {len(list(BALANCED_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
