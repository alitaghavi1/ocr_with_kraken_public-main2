"""
Prepare line-level training data for Kraken OCR fine-tuning.
Converts .txt files to .gt.txt format that Kraken expects.
"""

import os
import shutil
import glob
from pathlib import Path

# Source folders
SOURCES = [
    "training_data_lines/public_line_images",
    # Add more folders here as needed
]

# Output folder
OUTPUT_DIR = "prepared_line_training"


def prepare_data(sources, output_dir, copy_files=False):
    """
    Prepare training data by ensuring .gt.txt format.

    If copy_files=True: Copy files to output_dir
    If copy_files=False: Rename in place (faster, no duplication)
    """
    os.makedirs(output_dir, exist_ok=True)

    total_pairs = 0

    for source in sources:
        if not os.path.exists(source):
            print(f"Warning: {source} not found, skipping")
            continue

        print(f"\nProcessing: {source}")

        # Find all image files
        png_files = glob.glob(os.path.join(source, "*.png"))
        jpg_files = glob.glob(os.path.join(source, "*.jpg"))
        image_files = png_files + jpg_files

        valid = 0
        for img_path in image_files:
            img_name = Path(img_path).stem
            img_ext = Path(img_path).suffix
            img_dir = Path(img_path).parent

            # Look for ground truth file
            gt_txt = img_dir / f"{img_name}.txt"
            gt_gt_txt = img_dir / f"{img_name}.gt.txt"

            gt_source = None
            if gt_gt_txt.exists():
                gt_source = gt_gt_txt
            elif gt_txt.exists():
                gt_source = gt_txt
            else:
                continue  # No ground truth

            if copy_files:
                # Copy to output directory
                out_img = Path(output_dir) / f"{img_name}{img_ext}"
                out_gt = Path(output_dir) / f"{img_name}.gt.txt"

                if not out_img.exists():
                    shutil.copy2(img_path, out_img)
                if not out_gt.exists():
                    shutil.copy2(gt_source, out_gt)
            else:
                # Rename .txt to .gt.txt in place if needed
                if gt_source.suffix == '.txt' and not gt_gt_txt.exists():
                    os.rename(gt_source, gt_gt_txt)

            valid += 1

        print(f"  Valid pairs: {valid}")
        total_pairs += valid

    print(f"\n{'='*50}")
    print(f"Total training pairs ready: {total_pairs}")

    if copy_files:
        print(f"Output directory: {output_dir}")
    else:
        print("Files renamed in place (.txt -> .gt.txt)")

    return total_pairs


def count_existing():
    """Count existing training data"""
    for source in SOURCES:
        if os.path.exists(source):
            png = len(glob.glob(os.path.join(source, "*.png")))
            gt = len(glob.glob(os.path.join(source, "*.gt.txt")))
            txt = len(glob.glob(os.path.join(source, "*.txt")))
            print(f"{source}:")
            print(f"  PNG: {png}, .gt.txt: {gt}, .txt: {txt}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        count_existing()
    elif len(sys.argv) > 1 and sys.argv[1] == "copy":
        # Copy mode - creates new folder with all data
        prepare_data(SOURCES, OUTPUT_DIR, copy_files=True)
    else:
        # Default: rename in place (faster)
        print("Renaming .txt to .gt.txt in place...")
        prepare_data(SOURCES, OUTPUT_DIR, copy_files=False)
        print("\nTo run training:")
        print(f'  .venv\\Scripts\\ketos.exe -d cuda:0 train -o models/line_finetuned -f path -B 8 -N 50 --lag 10 -r 0.0001 --augment -i models/all_arabic_scripts.mlmodel --resize union "training_data_lines/public_line_images/*.png"')
