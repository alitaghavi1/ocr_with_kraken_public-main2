"""
Download British Library Arabic Manuscript Dataset (RASM2019)

Dataset: Ground Truth transcriptions for training OCR of historical Arabic handwritten texts
Source: https://bl.iro.bl.uk/concern/datasets/f866aefa-b025-4675-b37d-44647649ba71
License: CC Public Domain Mark 1.0

Contains:
- 120 high resolution (400dpi) TIFF images
- PAGE XML transcriptions
- Historical Arabic manuscripts (10th-19th century)
"""

import os
import requests
import zipfile
from pathlib import Path

# Download URLs (from British Library repository - correct URLs)
DATASET_URLS = {
    "RASM2019_part_1.zip": "https://bl.iro.bl.uk/downloads/e03280ef-5a75-4193-a8b5-1265f295e5cf?locale=en",
    "RASM2019_part_2.zip": "https://bl.iro.bl.uk/downloads/907b2e2a-3f23-49b8-8eef-f073c8bb97ab?locale=en",
}

OUTPUT_DIR = "training_data_lines/british_library_arabic"


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress indicator."""
    print(f"Downloading: {dest_path}")
    print(f"URL: {url}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
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
        if e.response.status_code == 429:
            print("  Rate limited. Please try again later or download manually.")
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


def main():
    print("=" * 60)
    print("British Library Arabic Manuscript Dataset Downloader")
    print("=" * 60)
    print()
    print("Dataset: RASM2019 - Historical Arabic Scientific Manuscripts")
    print("Size: ~8.7 GB total (2 ZIP files)")
    print("Content: 120 TIFF images + PAGE XML transcriptions")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download from British Library repository
    print("Downloading from British Library repository...")
    print()

    for filename, url in DATASET_URLS.items():
        dest_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(dest_path):
            print(f"Already exists: {dest_path}")
            continue

        success = download_file(url, dest_path)

        if success:
            # Extract the ZIP
            extract_zip(dest_path, OUTPUT_DIR)

    print()
    print("=" * 60)
    print("Download complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print()
    print("Next step: Run extract_bl_lines.py to prepare training data")
    print()
    print("If automatic download failed, manually download from:")
    print("  https://bl.iro.bl.uk/concern/datasets/f866aefa-b025-4675-b37d-44647649ba71")
    print("  or")
    print("  https://zenodo.org/record/3271395")


if __name__ == "__main__":
    main()
