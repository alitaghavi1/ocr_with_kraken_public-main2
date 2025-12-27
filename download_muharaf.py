"""
Download Muharaf Dataset - Manuscripts of Handwritten Arabic

Source: https://zenodo.org/records/11492215
Paper: https://arxiv.org/html/2406.09630v1
GitHub: https://github.com/mehreenmehreen/muharaf

Contains:
- 1,216 public manuscript images with line-level transcriptions
- Historical handwritten Arabic manuscripts (19th-21st century)
- Line images + ground truth text
"""

import os
import requests
import zipfile
from pathlib import Path

# Zenodo download URL
DATASET_URL = "https://zenodo.org/records/11492215/files/muharaf-public.zip?download=1"
OUTPUT_DIR = "training_data_lines/muharaf"
ZIP_FILE = "training_data_lines/muharaf/muharaf-public.zip"


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress indicator."""
    print(f"Downloading: {dest_path}")
    print(f"URL: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 / total_size
                    mb_down = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    print(f"\r  Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

        print(f"\n  Downloaded: {dest_path}")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"\n  HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    print(f"Extracting: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"  Extracted to: {extract_to}")
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def prepare_kraken_format():
    """Convert Muharaf data to Kraken training format."""
    import shutil
    import glob

    # Find all line images and their transcriptions
    muharaf_dir = Path(OUTPUT_DIR)
    output_lines_dir = Path("training_data_lines/muharaf_lines")
    output_lines_dir.mkdir(parents=True, exist_ok=True)

    # Look for line images - Muharaf stores them in specific structure
    # The structure is typically: images/ and transcriptions/

    total_pairs = 0

    # Search for PNG files with corresponding text files
    for png_file in muharaf_dir.rglob("*.png"):
        # Try different possible locations for ground truth
        possible_gt = [
            png_file.with_suffix('.txt'),
            png_file.with_suffix('.gt.txt'),
            png_file.parent / (png_file.stem + '.txt'),
        ]

        gt_file = None
        for gt_path in possible_gt:
            if gt_path.exists():
                gt_file = gt_path
                break

        if gt_file:
            # Create unique output name
            out_name = f"muharaf_{total_pairs:05d}"
            out_png = output_lines_dir / f"{out_name}.png"
            out_gt = output_lines_dir / f"{out_name}.gt.txt"

            try:
                shutil.copy2(png_file, out_png)
                shutil.copy2(gt_file, out_gt)
                total_pairs += 1
            except Exception as e:
                print(f"Error copying {png_file.name}: {e}")

    # Also check for JPG files
    for jpg_file in muharaf_dir.rglob("*.jpg"):
        possible_gt = [
            jpg_file.with_suffix('.txt'),
            jpg_file.with_suffix('.gt.txt'),
        ]

        gt_file = None
        for gt_path in possible_gt:
            if gt_path.exists():
                gt_file = gt_path
                break

        if gt_file:
            out_name = f"muharaf_{total_pairs:05d}"
            out_png = output_lines_dir / f"{out_name}.png"
            out_gt = output_lines_dir / f"{out_name}.gt.txt"

            try:
                # Convert JPG to PNG for consistency
                from PIL import Image
                img = Image.open(jpg_file)
                img.save(out_png)
                shutil.copy2(gt_file, out_gt)
                total_pairs += 1
            except Exception as e:
                print(f"Error processing {jpg_file.name}: {e}")

    return total_pairs


def main():
    print("=" * 60)
    print("Muharaf Dataset Downloader")
    print("Manuscripts of Handwritten Arabic")
    print("=" * 60)
    print()
    print("Source: Zenodo (https://zenodo.org/records/11492215)")
    print("Content: Historical handwritten Arabic manuscripts")
    print("         Line images with transcriptions")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download if not exists
    if not os.path.exists(ZIP_FILE):
        if not download_file(DATASET_URL, ZIP_FILE):
            print()
            print("Download failed. Please manually download from:")
            print("  https://zenodo.org/records/11492215")
            return
    else:
        print(f"ZIP file already exists: {ZIP_FILE}")

    # Extract
    extract_zip(ZIP_FILE, OUTPUT_DIR)

    # List contents
    print()
    print("Exploring downloaded content...")
    for item in Path(OUTPUT_DIR).iterdir():
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/: {count} files")
        else:
            print(f"  {item.name}")

    print()
    print("=" * 60)
    print("Download complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print()
    print("Note: Run prepare_combined_training.py to prepare for training")


if __name__ == "__main__":
    main()
