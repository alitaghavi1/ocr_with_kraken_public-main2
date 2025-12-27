"""
Create a balanced training dataset for Kraken fine-tuning.
- Includes: public_line_images, bl_extracted_lines, khatt_lines
- Converts all images to grayscale
- Never auto-deletes existing data
"""

import random
import shutil
from pathlib import Path
from PIL import Image

# Paths
BASE_DIR = Path(__file__).parent
PUBLIC_LINES = BASE_DIR / "training_data_lines" / "Muharaf_public_line_images"
BL_LINES = BASE_DIR / "training_data_lines" / "bl_extracted_lines"
BL_RESIZED = BASE_DIR / "training_data_lines" / "bl_resized"  # Cache resized images
KHATT_LINES = BASE_DIR / "training_data_lines" / "khatt_lines"
BALANCED_DIR = BASE_DIR / "training_data_lines" / "balanced_training"

# Settings
RANDOM_SEED = 42
TARGET_HEIGHT = 64


def resize_image(src_path, dest_path, target_height):
    """Resize image to target height, maintaining aspect ratio."""
    img = Image.open(src_path)
    if img.height > target_height:
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
    img.save(dest_path)


def ensure_bl_resized():
    """Resize BL images once and cache them."""
    bl_files = list(BL_LINES.glob("*.png"))

    # Check if already resized
    if BL_RESIZED.exists():
        existing = len(list(BL_RESIZED.glob("*.png")))
        if existing == len(bl_files):
            print(f"Using cached resized BL images ({existing} files)")
            return list(BL_RESIZED.glob("*.png"))

    # Resize and cache
    print(f"Resizing {len(bl_files)} BL images (one-time operation)...")
    BL_RESIZED.mkdir(parents=True, exist_ok=True)

    for png_path in bl_files:
        dest_png = BL_RESIZED / png_path.name
        resize_image(png_path, dest_png, TARGET_HEIGHT)

        # Copy ground truth
        gt_path = png_path.with_suffix('.gt.txt')
        if gt_path.exists():
            shutil.copy2(gt_path, BL_RESIZED / gt_path.name)

    print(f"  Resized images cached to: {BL_RESIZED}")
    return list(BL_RESIZED.glob("*.png"))


def has_valid_gt(png_path):
    """Check if image has non-empty ground truth."""
    gt_path = png_path.with_suffix('.gt.txt')
    if not gt_path.exists():
        return False
    try:
        text = gt_path.read_text(encoding='utf-8').strip()
        return len(text) > 0
    except:
        return False


def copy_as_grayscale(src_path, dest_path):
    """Copy image, converting to grayscale."""
    img = Image.open(src_path)
    if img.mode != 'L':
        img = img.convert('L')
    img.save(dest_path)


def main():
    random.seed(RANDOM_SEED)

    # Step 1: Ensure BL images are resized (cached)
    bl_files = ensure_bl_resized()
    bl_files = [f for f in bl_files if has_valid_gt(f)]
    print(f"bl_extracted_lines: {len(bl_files)} files (with valid GT)")

    # Step 2: Get all public_line_images (filter empty GT)
    public_files = [f for f in PUBLIC_LINES.glob("*.png") if has_valid_gt(f)]
    print(f"Muharaf_public_line_images: {len(public_files)} files with valid GT")

    # Step 3: Get KHATT files
    khatt_files = []
    if KHATT_LINES.exists():
        khatt_files = [f for f in KHATT_LINES.glob("*.png") if has_valid_gt(f)]
        print(f"khatt_lines: {len(khatt_files)} files with valid GT")
    else:
        print(f"khatt_lines: not found (skipping)")

    expected_total = len(bl_files) + len(public_files) + len(khatt_files)
    print(f"\nExpected total: {expected_total}")

    # Step 4: Check if balanced_training already exists - NEVER auto-delete
    if BALANCED_DIR.exists():
        existing = len(list(BALANCED_DIR.glob("*.png")))
        if existing >= expected_total:
            print(f"\nBalanced dataset already exists ({existing} files)")
            print("Delete 'balanced_training' folder manually to regenerate")
            return
        else:
            print(f"\nExisting balanced_training has {existing} files, expected {expected_total}")
            print("Removing old data and regenerating...")
            shutil.rmtree(BALANCED_DIR)

    BALANCED_DIR.mkdir(parents=True)

    # Step 5: Copy files with grayscale conversion
    print(f"\nCreating balanced dataset (converting to grayscale)...")
    total = 0

    # Copy BL files
    print("  Copying bl_extracted_lines...")
    for i, png_path in enumerate(bl_files):
        dest_png = BALANCED_DIR / f"line_{total:05d}.png"
        copy_as_grayscale(png_path, dest_png)

        gt_path = png_path.with_suffix('.gt.txt')
        if gt_path.exists():
            shutil.copy2(gt_path, BALANCED_DIR / f"line_{total:05d}.gt.txt")
        total += 1

    # Copy Muharaf files
    print("  Copying Muharaf_public_line_images...")
    for j, png_path in enumerate(public_files):
        dest_png = BALANCED_DIR / f"line_{total:05d}.png"
        copy_as_grayscale(png_path, dest_png)

        gt_path = png_path.with_suffix('.gt.txt')
        if gt_path.exists():
            shutil.copy2(gt_path, BALANCED_DIR / f"line_{total:05d}.gt.txt")
        total += 1

        if (j + 1) % 5000 == 0:
            print(f"    {j + 1}/{len(public_files)}...")

    # Copy KHATT files
    if khatt_files:
        print("  Copying khatt_lines...")
        for k, png_path in enumerate(khatt_files):
            dest_png = BALANCED_DIR / f"line_{total:05d}.png"
            copy_as_grayscale(png_path, dest_png)

            gt_path = png_path.with_suffix('.gt.txt')
            if gt_path.exists():
                shutil.copy2(gt_path, BALANCED_DIR / f"line_{total:05d}.gt.txt")
            total += 1

            if (k + 1) % 2000 == 0:
                print(f"    {k + 1}/{len(khatt_files)}...")

    print(f"\n{'='*50}")
    print(f"Total files: {total}")
    print(f"Balanced dataset: {BALANCED_DIR}")
    print(f"All images converted to grayscale")


if __name__ == "__main__":
    main()
