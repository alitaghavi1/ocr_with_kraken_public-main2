"""
Download Persian dictionary for OCR post-processing.

Sources:
- shahind/Persian-Words-Database (~700K words)
- Hazm stopwords
"""

import urllib.request
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DICT_DIR = BASE_DIR / "dictionaries"

SOURCES = {
    "persian_words": "https://raw.githubusercontent.com/shahind/Persian-Words-Database/master/distinct_words.txt",
    "persian_big": "https://raw.githubusercontent.com/shahind/Persian-Words-Database/master/big.txt",
}


def download_file(url, dest_path):
    """Download a file from URL."""
    print(f"Downloading: {url}")
    print(f"  -> {dest_path}")

    try:
        urllib.request.urlretrieve(url, dest_path)

        # Count lines
        with open(dest_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
        print(f"  Downloaded: {lines:,} lines")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading Persian Dictionary Files")
    print("=" * 60)

    DICT_DIR.mkdir(exist_ok=True)

    for name, url in SOURCES.items():
        dest = DICT_DIR / f"{name}.txt"
        download_file(url, dest)

    # Create combined dictionary
    print("\nCreating combined dictionary...")
    combined = set()

    for txt_file in DICT_DIR.glob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    word = line.strip()
                    # Filter: Persian words, 2+ characters
                    if word and len(word) >= 2:
                        combined.add(word)
        except:
            pass

    # Save combined
    combined_path = BASE_DIR / "persian_dictionary_ganjoor.txt"
    with open(combined_path, 'w', encoding='utf-8') as f:
        for word in sorted(combined):
            f.write(word + '\n')

    print(f"\nCombined dictionary: {len(combined):,} unique words")
    print(f"Saved to: {combined_path}")

    # Show sample
    print("\nSample words:")
    sample = list(combined)[:20]
    for w in sample:
        print(f"  {w}")


if __name__ == "__main__":
    main()
