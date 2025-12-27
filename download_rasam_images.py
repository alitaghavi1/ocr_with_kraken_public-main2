"""
Download all RASAM images from BULAC IIIF server.
Uses multiple threads for faster downloading.
"""

import urllib.request
import ssl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"
IMAGES_DIR = BASE_DIR / "training_data_lines" / "rasam_images"

IIIF_BASE = "https://bina.bulac.fr/iiif/2"

# Create SSL context that doesn't verify certificates
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def download_image(image_id, width=2000):
    """Download single image from IIIF server."""
    dest = IMAGES_DIR / f"{image_id}.jpg"

    if dest.exists():
        return image_id, "exists"

    url = f"{IIIF_BASE}/{image_id}/full/{width},/0/default.jpg"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        return image_id, "ok"
    except Exception as e:
        return image_id, f"error: {e}"


def load_image_mapping():
    """Load mapping from filename to IIIF image ID from TSV."""
    tsv_path = REPO_DIR / "list-images.tsv"
    mapping = {}

    with open(tsv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                filename = parts[3]  # FileName column
                iiif_id = parts[6]   # IIIF image ID column
                mapping[filename] = iiif_id

    return mapping


def download_image_by_mapping(filename, iiif_id, width=2000):
    """Download image using IIIF ID, save with filename."""
    dest = IMAGES_DIR / f"{filename}.jpg"

    if dest.exists():
        return filename, "exists"

    url = f"{IIIF_BASE}/{iiif_id}/full/{width},/0/default.jpg"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        return filename, "ok"
    except Exception as e:
        return filename, f"error: {e}"


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Image Downloader")
    print("="*50)

    # Load filename -> IIIF ID mapping
    print("Loading image mapping from TSV...")
    mapping = load_image_mapping()
    print(f"Found {len(mapping)} images in mapping\n")

    # Get XML files and filter to those with mapping
    xml_files = list(REPO_DIR.glob("page/**/*.xml"))
    to_download = [(xml.stem, mapping[xml.stem]) for xml in xml_files if xml.stem in mapping]

    print(f"XML files: {len(xml_files)}")
    print(f"With IIIF mapping: {len(to_download)}")
    print(f"Output: {IMAGES_DIR}")
    print(f"Using 4 parallel threads\n")

    # Check how many already exist
    existing = sum(1 for fn, _ in to_download if (IMAGES_DIR / f"{fn}.jpg").exists())
    print(f"Already downloaded: {existing}")
    print(f"Remaining: {len(to_download) - existing}\n")

    if existing == len(to_download):
        print("All images already downloaded!")
        return

    # Download with thread pool
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_image_by_mapping, fn, iiif_id): fn
                   for fn, iiif_id in to_download}

        for future in as_completed(futures):
            fn = futures[future]
            result_fn, status = future.result()

            if status == "ok":
                completed += 1
                if completed % 20 == 0:
                    print(f"  Downloaded: {completed}")
            elif status == "exists":
                pass
            else:
                errors += 1
                print(f"  Failed: {result_fn} - {status}")

    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  Downloaded: {completed}")
    print(f"  Already existed: {existing}")
    print(f"  Errors: {errors}")
    print(f"\nNow run: .venv\\Scripts\\python.exe extract_rasam_lines_local.py")


if __name__ == "__main__":
    main()
