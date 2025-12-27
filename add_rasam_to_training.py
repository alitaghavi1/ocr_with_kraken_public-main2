"""
Add RASAM lines to balanced_training folder.
Converts to grayscale and copies with proper numbering.
"""

from pathlib import Path
from PIL import Image
import shutil

BASE_DIR = Path(__file__).parent
RASAM_DIR = BASE_DIR / "training_data_lines" / "rasam_lines"
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"

def get_next_index(output_dir):
    """Find the highest existing index in output directory."""
    existing = list(output_dir.glob("*.png"))
    if not existing:
        return 0

    max_idx = 0
    for f in existing:
        try:
            # Extract number from filename like "line_00001.png" or "rasam_00001.png"
            name = f.stem
            # Try to get the number part
            parts = name.split('_')
            for part in parts:
                if part.isdigit():
                    max_idx = max(max_idx, int(part))
                    break
        except:
            pass
    return max_idx + 1

def main():
    print("Adding RASAM lines to balanced_training")
    print("=" * 50)

    # Get RASAM files
    rasam_pngs = sorted(RASAM_DIR.glob("*.png"))
    print(f"RASAM lines to add: {len(rasam_pngs)}")

    # Check existing balanced_training
    existing = list(BALANCED_DIR.glob("*.png"))
    print(f"Existing balanced_training: {len(existing)}")

    # Find starting index
    start_idx = get_next_index(BALANCED_DIR)
    print(f"Starting index: {start_idx}")

    # Copy files
    copied = 0
    skipped = 0

    for png_path in rasam_pngs:
        gt_path = png_path.with_suffix('.gt.txt')

        if not gt_path.exists():
            skipped += 1
            continue

        # New names
        new_png = BALANCED_DIR / f"rasam_{start_idx + copied:05d}.png"
        new_gt = BALANCED_DIR / f"rasam_{start_idx + copied:05d}.gt.txt"

        # Skip if already exists
        if new_png.exists():
            skipped += 1
            continue

        try:
            # Convert to grayscale
            img = Image.open(png_path)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(new_png)

            # Copy ground truth
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
