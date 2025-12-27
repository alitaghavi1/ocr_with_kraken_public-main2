"""
Download OpenITI Gold Standard OCR Training Data

Source: https://github.com/OpenITI/OCR_GS_Data
License: CC BY-NC-SA 4.0 (non-commercial)

Contains:
- 7,000+ double-checked line images with transcriptions
- Arabic, Persian, Urdu, Azerbaijani Turkish data
- Already in Kraken format (.png + .gt.txt pairs)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
REPO_URL = "https://github.com/OpenITI/OCR_GS_Data.git"
CLONE_DIR = "training_data_lines/openiti_raw"
OUTPUT_DIR = "training_data_lines/openiti_gs_lines"

# Languages to include (subdirectories in the repo)
LANGUAGES = ["ara", "fas"]  # Arabic and Persian


def clone_repo():
    """Clone the OpenITI OCR_GS_Data repository."""
    if os.path.exists(CLONE_DIR):
        print(f"Repository already exists at {CLONE_DIR}")
        return True

    print(f"Cloning OpenITI OCR_GS_Data repository...")
    print(f"URL: {REPO_URL}")
    print()

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, CLONE_DIR],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Git clone failed: {result.stderr}")
            return False
        print(f"Cloned to: {CLONE_DIR}")
        return True
    except FileNotFoundError:
        print("Git not found. Please install Git or download manually from:")
        print(f"  {REPO_URL}")
        return False
    except Exception as e:
        print(f"Error cloning: {e}")
        return False


def collect_training_pairs():
    """Collect all .png/.gt.txt pairs from the cloned repository."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_pairs = 0

    for lang in LANGUAGES:
        lang_dir = os.path.join(CLONE_DIR, lang)
        if not os.path.exists(lang_dir):
            print(f"Language directory not found: {lang_dir}")
            continue

        print(f"\nProcessing {lang} data...")
        lang_pairs = 0

        # Walk through all subdirectories
        for root, dirs, files in os.walk(lang_dir):
            png_files = [f for f in files if f.endswith('.png')]

            for png_file in png_files:
                # Check for corresponding .gt.txt file
                base_name = png_file[:-4]  # Remove .png
                gt_file = base_name + '.gt.txt'

                png_path = os.path.join(root, png_file)
                gt_path = os.path.join(root, gt_file)

                if os.path.exists(gt_path):
                    # Create unique output name
                    rel_path = os.path.relpath(root, lang_dir)
                    safe_name = rel_path.replace(os.sep, '_').replace(' ', '_')
                    out_name = f"openiti_{lang}_{safe_name}_{base_name}"

                    # Copy files to output directory
                    out_png = os.path.join(OUTPUT_DIR, out_name + '.png')
                    out_gt = os.path.join(OUTPUT_DIR, out_name + '.gt.txt')

                    try:
                        shutil.copy2(png_path, out_png)
                        shutil.copy2(gt_path, out_gt)
                        lang_pairs += 1
                    except Exception as e:
                        print(f"  Error copying {png_file}: {e}")

        print(f"  Found {lang_pairs} line pairs for {lang}")
        total_pairs += lang_pairs

    return total_pairs


def verify_data():
    """Verify the collected training data."""
    if not os.path.exists(OUTPUT_DIR):
        return 0, 0

    png_files = list(Path(OUTPUT_DIR).glob("*.png"))
    gt_files = list(Path(OUTPUT_DIR).glob("*.gt.txt"))

    # Check for matching pairs
    pairs = 0
    for png in png_files:
        gt = png.with_suffix('.gt.txt')
        if gt.exists():
            pairs += 1

    return len(png_files), pairs


def main():
    print("=" * 60)
    print("OpenITI Gold Standard OCR Data Downloader")
    print("=" * 60)
    print()
    print("Source: GitHub OpenITI/OCR_GS_Data")
    print("License: CC BY-NC-SA 4.0 (non-commercial use)")
    print("Content: Line images + transcriptions for Arabic/Persian")
    print()

    # Step 1: Clone repository
    if not clone_repo():
        print()
        print("Failed to clone repository.")
        print("Please manually clone or download from:")
        print(f"  {REPO_URL}")
        return

    # Step 2: Collect training pairs
    print()
    print("Collecting training data pairs...")
    total_pairs = collect_training_pairs()

    # Step 3: Verify
    png_count, pair_count = verify_data()

    print()
    print("=" * 60)
    print("Download and preparation complete!")
    print(f"  Total images: {png_count}")
    print(f"  Valid pairs: {pair_count}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    print("Data is ready for Kraken training.")
    print(f'  Pattern: "{OUTPUT_DIR}/*.png"')


if __name__ == "__main__":
    main()
