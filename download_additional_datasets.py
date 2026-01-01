"""
Additional Arabic/Persian OCR Datasets Guide

This script provides information on additional datasets that can be used
for training Arabic and Persian OCR models.

Run this script to see dataset information and download instructions.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    print_section("ADDITIONAL ARABIC/PERSIAN OCR DATASETS")

    # =========================================================================
    # ARABIC DATASETS
    # =========================================================================
    print_section("ARABIC DATASETS - FREELY AVAILABLE")

    datasets_arabic = [
        {
            "name": "1. VML-AHTE (Arabic Handwritten Text Line Extraction)",
            "lines": "~500 lines (30 pages)",
            "source": "GitHub",
            "url": "https://github.com/beratkurar/arabic_handwritten_textline_extraction_dataset",
            "format": "PAGE XML + Images",
            "script": "download_vml_ahte.py",
            "notes": "Native Arabic speaker annotations, crowded diacritics"
        },
        {
            "name": "2. Historical Arabic Handwritten Text Recognition",
            "lines": "40 pages (expert transcribed)",
            "source": "Mendeley Data",
            "url": "https://data.mendeley.com/datasets/xz6f8bw3w8/1",
            "format": "Images + Text",
            "script": "Manual download",
            "notes": "Ancient manuscripts from Islamic University of Madinah"
        },
        {
            "name": "3. Muharaf Dataset (Already have script)",
            "lines": "36,000+ lines",
            "source": "Zenodo",
            "url": "https://zenodo.org/records/11492215",
            "format": "JSON annotations + Images",
            "script": "download_muharaf.py",
            "notes": "Historical manuscripts 19th-21st century"
        },
        {
            "name": "4. KHATT Dataset (Already have script)",
            "lines": "6,600+ lines",
            "source": "Hugging Face",
            "url": "https://huggingface.co/datasets/johnlockejrr/KHATT_v1.0_dataset",
            "format": "Images + Ground Truth",
            "script": "download_khatt.py",
            "notes": "Arabic handwriting samples"
        },
        {
            "name": "5. RASAM Dataset (Already have script)",
            "lines": "11,290 lines",
            "source": "GitHub",
            "url": "https://github.com/calfa-co/rasam-dataset",
            "format": "PAGE XML + IIIF images",
            "script": "download_rasam.py",
            "notes": "Maghrebi Arabic manuscripts"
        },
        {
            "name": "6. OpenITI Gold Standard (Already have script)",
            "lines": "7,000+ lines",
            "source": "GitHub",
            "url": "https://github.com/OpenITI/OCR_GS_Data",
            "format": "PNG + GT.TXT (Kraken ready)",
            "script": "download_openiti_data.py",
            "notes": "Arabic, Persian, Urdu - already in Kraken format!"
        },
        {
            "name": "7. Arabic Documents OCR Dataset",
            "lines": "Varies",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset",
            "format": "Images + Annotations",
            "script": "Manual download (requires Kaggle account)",
            "notes": "Modern Arabic documents"
        },
        {
            "name": "8. Yarmouk OCR Dataset",
            "lines": "Varies",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/eyadwin/yarmouk-ocr-dataset",
            "format": "Images + Text",
            "script": "Manual download (requires Kaggle account)",
            "notes": "Arabic text samples"
        },
    ]

    for ds in datasets_arabic:
        print(f"\n{ds['name']}")
        print(f"  Lines: {ds['lines']}")
        print(f"  Source: {ds['source']}")
        print(f"  URL: {ds['url']}")
        print(f"  Format: {ds['format']}")
        print(f"  Script: {ds['script']}")
        print(f"  Notes: {ds['notes']}")

    # =========================================================================
    # PERSIAN DATASETS
    # =========================================================================
    print_section("PERSIAN/FARSI DATASETS")

    datasets_persian = [
        {
            "name": "1. OpenITI Persian (Already have script)",
            "lines": "Part of OpenITI GS",
            "source": "GitHub",
            "url": "https://github.com/OpenITI/OCR_GS_Data",
            "format": "PNG + GT.TXT (Kraken ready)",
            "script": "download_openiti_data.py",
            "notes": "Persian (fas) subdirectory - already Kraken format!"
        },
        {
            "name": "2. PHTD (Persian Handwritten Text Dataset)",
            "lines": "1,787 lines",
            "source": "ResearchGate (paper)",
            "url": "https://www.researchgate.net/publication/254018620",
            "format": "Images + Pixel/Content GT",
            "script": "Contact authors",
            "notes": "140 documents, 40 writers, text-line level"
        },
        {
            "name": "3. Concordia Persian Database",
            "lines": "500 writers",
            "source": "Concordia University",
            "url": "https://users.encs.concordia.ca/~j_sadri/PersianDatabase.htm",
            "format": "Various",
            "script": "download_concordia_persian.py (sample)",
            "notes": "Free sample available, full requires request"
        },
        {
            "name": "4. PHCWT (Persian Handwritten Characters Words Text)",
            "lines": "400 texts",
            "source": "IEEE",
            "url": "https://ieeexplore.ieee.org/document/8342357",
            "format": "Character/Word/Text levels",
            "script": "Contact authors",
            "notes": "51,200 chars, 3,600 words, 400 texts"
        },
        {
            "name": "5. Ganjoor Persian Poetry",
            "lines": "Varies",
            "source": "Ganjoor",
            "url": "https://ganjoor.net/",
            "format": "Text corpus (no images)",
            "script": "download_ganjoor.py",
            "notes": "Text only - useful for synthetic data generation"
        },
    ]

    for ds in datasets_persian:
        print(f"\n{ds['name']}")
        print(f"  Lines: {ds['lines']}")
        print(f"  Source: {ds['source']}")
        print(f"  URL: {ds['url']}")
        print(f"  Format: {ds['format']}")
        print(f"  Script: {ds['script']}")
        print(f"  Notes: {ds['notes']}")

    # =========================================================================
    # RECOMMENDED DOWNLOAD ORDER
    # =========================================================================
    print_section("RECOMMENDED DOWNLOAD ORDER")

    print("""
For maximum training data, run these scripts in order:

ARABIC:
  1. python download_openiti_data.py     # 7,000+ lines (Kraken-ready!)
  2. python download_muharaf.py          # 36,000+ lines
  3. python download_khatt.py            # 6,600+ lines
  4. python download_rasam.py            # 11,290 lines
  5. python download_vml_ahte.py         # ~500 lines

PERSIAN:
  1. python download_openiti_data.py     # Includes Persian (fas)
  2. python download_concordia_persian.py # Sample only

HUGGING FACE (requires pip install huggingface_hub datasets):
  1. python download_hf_arabic_ocr.py    # Various Arabic datasets

TOTAL POTENTIAL: ~60,000+ line images
""")

    # =========================================================================
    # QUICK DOWNLOAD ALL
    # =========================================================================
    print_section("QUICK COMMANDS")

    print("""
# Download all available datasets (run from project directory):

# Step 1: Install dependencies
pip install huggingface_hub datasets requests pillow

# Step 2: Run downloaders
python download_openiti_data.py
python download_muharaf.py
python download_khatt.py
python download_vml_ahte.py
python download_hf_arabic_ocr.py

# Step 3: Create balanced training manifest
python create_balanced_manifest.py

# Step 4: Start training
python train.py
""")

    # =========================================================================
    # DATASETS REQUIRING MANUAL ACTION
    # =========================================================================
    print_section("DATASETS REQUIRING MANUAL DOWNLOAD")

    print("""
These datasets require manual download or registration:

1. MENDELEY HISTORICAL ARABIC
   URL: https://data.mendeley.com/datasets/xz6f8bw3w8/1
   Action: Download ZIP manually, extract to training_data_lines/mendeley_arabic/

2. KAGGLE DATASETS (requires Kaggle account)
   - Arabic Documents OCR: https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset
   - Yarmouk OCR: https://www.kaggle.com/datasets/eyadwin/yarmouk-ocr-dataset
   Action: kaggle datasets download <dataset-name>

3. PHTD PERSIAN
   Contact authors through ResearchGate for access

4. PHCWT PERSIAN
   Contact authors through IEEE for access

5. CONCORDIA FULL DATABASE
   Contact: j_sadri@encs.concordia.ca
""")


if __name__ == "__main__":
    main()
