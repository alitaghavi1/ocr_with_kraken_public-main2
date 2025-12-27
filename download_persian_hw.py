"""
Download and explore Persian Handwriting dataset from GitHub.
"""

import urllib.request
from pathlib import Path
import scipy.io as sio
import numpy as np

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "persian_hw"

# GitHub raw URLs
GITHUB_BASE = "https://github.com/myousefnezhad/persianhandwriting/raw/master"

FILES = [
    "PersianHWR1.mat",
    "Class.mat",
]

def download_file(url, dest):
    """Download file from URL."""
    print(f"  Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"    Done: {dest.stat().st_size / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"    Error: {e}")
        return False

def explore_mat_file(mat_path):
    """Explore contents of a MAT file."""
    print(f"\nExploring: {mat_path.name}")
    try:
        data = sio.loadmat(mat_path)
        print(f"  Keys: {[k for k in data.keys() if not k.startswith('__')]}")

        for key in data.keys():
            if key.startswith('__'):
                continue
            val = data[key]
            if isinstance(val, np.ndarray):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                # Show sample if small
                if val.size < 50:
                    print(f"    Sample: {val.flatten()[:20]}")
            else:
                print(f"  {key}: {type(val)}")
        return data
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Persian Handwriting dataset...")

    # Download MAT files
    for filename in FILES:
        url = f"{GITHUB_BASE}/{filename}"
        dest = OUTPUT_DIR / filename
        if not dest.exists():
            download_file(url, dest)
        else:
            print(f"  {filename} already exists")

    # Explore the files
    print("\n" + "="*50)
    print("Exploring dataset structure...")

    for filename in FILES:
        mat_path = OUTPUT_DIR / filename
        if mat_path.exists():
            explore_mat_file(mat_path)

    print("\n" + "="*50)
    print("Note: This dataset contains Persian LETTERS (characters),")
    print("not full text lines. It may not be ideal for Kraken training.")
    print("Kraken works best with LINE-level images + transcriptions.")

if __name__ == "__main__":
    main()
