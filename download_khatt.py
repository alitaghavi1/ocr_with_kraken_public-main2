"""
Download KHATT dataset from Hugging Face and prepare for Kraken training.
Uses huggingface_hub to download files directly.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
import tarfile
import os
import shutil

# Paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "khatt_lines"
TEMP_DIR = BASE_DIR / "temp_khatt"

REPO_ID = "johnlockejrr/KHATT_v1.0_dataset"

def parse_ground_truth_file(gt_file_path):
    """Parse ground truth file and return dict mapping image names to text."""
    gt_map = {}
    try:
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format is typically: image_path text or image_path\ttext
                if '\t' in line:
                    parts = line.split('\t', 1)
                elif ' ' in line:
                    parts = line.split(' ', 1)
                else:
                    continue

                if len(parts) == 2:
                    img_name = Path(parts[0]).stem  # Get filename without extension
                    text = parts[1].strip()
                    if text:
                        gt_map[img_name] = text
    except Exception as e:
        print(f"  Error parsing {gt_file_path}: {e}")
    return gt_map

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading KHATT dataset from Hugging Face...")
    print(f"Repository: {REPO_ID}")
    print("This may take a few minutes...\n")

    # Step 1: Download config_files.tar.bz2 (contains ground truth)
    print("Step 1: Downloading ground truth (config_files.tar.bz2)...")
    try:
        config_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="config_files.tar.bz2",
            repo_type="dataset",
            local_dir=TEMP_DIR
        )
        print(f"  Downloaded: {config_path}")

        # Extract
        print("  Extracting...")
        with tarfile.open(config_path, 'r:bz2') as tar:
            tar.extractall(TEMP_DIR)
        print("  Extracted config files")
    except Exception as e:
        print(f"  Error downloading config: {e}")
        return

    # Step 2: Find and parse ground truth files
    print("\nStep 2: Parsing ground truth files...")
    gt_map = {}

    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            if file.endswith('.txt') and ('train' in file.lower() or 'test' in file.lower() or 'val' in file.lower() or 'gt' in file.lower()):
                gt_file = Path(root) / file
                print(f"  Found: {gt_file.name}")
                parsed = parse_ground_truth_file(gt_file)
                gt_map.update(parsed)
                print(f"    Loaded {len(parsed)} entries")

    print(f"  Total ground truth entries: {len(gt_map)}")

    if not gt_map:
        # Try looking at the structure of config files
        print("\n  Exploring config file structure...")
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files[:20]:
                print(f"    {Path(root).name}/{file}")

    # Step 3: Download images and match with ground truth
    print("\nStep 3: Downloading images...")

    files = list_repo_files(REPO_ID, repo_type="dataset")
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and 'data/' in f]
    print(f"  Found {len(img_files)} images to download")

    total_saved = 0
    total_skipped = 0

    for i, img_file in enumerate(img_files):
        img_stem = Path(img_file).stem

        # Check if we have ground truth for this image
        if img_stem not in gt_map:
            total_skipped += 1
            continue

        # Download image
        try:
            local_img = hf_hub_download(
                repo_id=REPO_ID,
                filename=img_file,
                repo_type="dataset",
                local_dir=TEMP_DIR
            )

            # Save to output
            out_img = OUTPUT_DIR / f"khatt_{total_saved:05d}.png"
            out_gt = OUTPUT_DIR / f"khatt_{total_saved:05d}.gt.txt"

            img = Image.open(local_img)
            if img.mode != 'L':
                img = img.convert('L')
            img.save(out_img)
            out_gt.write_text(gt_map[img_stem], encoding='utf-8')

            total_saved += 1

            if total_saved % 500 == 0:
                print(f"  Saved {total_saved} images...")

        except Exception as e:
            print(f"  Error with {img_file}: {e}")

    # Cleanup
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print(f"\n{'='*50}")
    print(f"KHATT dataset download complete!")
    print(f"Total line images saved: {total_saved}")
    print(f"Skipped (no ground truth): {total_skipped}")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
