"""
Extract line images from Arshasb_7k dataset.

This script:
1. Reads each page image
2. Uses line coordinates from Excel to crop lines
3. Gets text from fulltext file
4. Saves as image + .gt.txt pairs for Kraken training
"""

import os
import re
import glob
from pathlib import Path
from PIL import Image
import pandas as pd

# Configuration
SOURCE_DIR = "training_data_lines/Arshasb_7k"
OUTPUT_DIR = "training_data_lines/Arshasb_extracted"
PADDING = 5  # Extra pixels around line crop


def parse_point(point_str):
    """Parse point string like '(2, 17)' to tuple (2, 17)"""
    if isinstance(point_str, tuple):
        return point_str
    match = re.match(r'\((\d+),\s*(\d+)\)', str(point_str))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def extract_lines_from_page(page_folder, output_dir, page_num):
    """Extract all lines from a single page."""
    folder_name = os.path.basename(page_folder)
    num = folder_name  # e.g., "00001"

    # Find files
    page_img = os.path.join(page_folder, f"page_{num}.png")
    line_xlsx = os.path.join(page_folder, f"line_{num}.xlsx")
    fulltext = os.path.join(page_folder, f"fulltext_{num}.txt")

    if not all(os.path.exists(f) for f in [page_img, line_xlsx, fulltext]):
        print(f"  Missing files in {folder_name}, skipping")
        return 0

    # Load page image
    try:
        img = Image.open(page_img)
    except Exception as e:
        print(f"  Error loading image: {e}")
        return 0

    # Load line coordinates
    try:
        df = pd.read_excel(line_xlsx)
    except Exception as e:
        print(f"  Error loading Excel: {e}")
        return 0

    # Load fulltext (line-by-line transcription)
    try:
        with open(fulltext, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
        # Skip first line (page number) and strip whitespace
        text_lines = [line.strip() for line in text_lines[1:] if line.strip()]
    except Exception as e:
        print(f"  Error loading fulltext: {e}")
        return 0

    extracted = 0

    for idx, row in df.iterrows():
        line_num = row.get('line', idx + 1)

        # Get text for this line
        if idx >= len(text_lines):
            continue
        text = text_lines[idx]

        if not text or len(text) < 2:
            continue

        # Parse coordinates
        p1 = parse_point(row['point1'])
        p2 = parse_point(row['point2'])
        p3 = parse_point(row['point3'])
        p4 = parse_point(row['point4'])

        if not all([p1, p2, p3, p4]):
            continue

        # Calculate bounding box
        all_x = [p1[0], p2[0], p3[0], p4[0]]
        all_y = [p1[1], p2[1], p3[1], p4[1]]

        left = max(0, min(all_x) - PADDING)
        top = max(0, min(all_y) - PADDING)
        right = min(img.width, max(all_x) + PADDING)
        bottom = min(img.height, max(all_y) + PADDING)

        # Crop line
        try:
            line_img = img.crop((left, top, right, bottom))
        except Exception as e:
            continue

        # Skip very small crops
        if line_img.width < 50 or line_img.height < 10:
            continue

        # Save files
        out_name = f"arshasb_{num}_line{line_num:03d}"
        out_img_path = os.path.join(output_dir, f"{out_name}.png")
        out_gt_path = os.path.join(output_dir, f"{out_name}.gt.txt")

        line_img.save(out_img_path)
        with open(out_gt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        extracted += 1

    return extracted


def main():
    print("="*60)
    print("Extracting lines from Arshasb_7k dataset")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all page folders
    page_folders = sorted(glob.glob(os.path.join(SOURCE_DIR, "*")))
    page_folders = [f for f in page_folders if os.path.isdir(f)]

    print(f"Found {len(page_folders)} pages to process")
    print(f"Output: {OUTPUT_DIR}")
    print()

    total_lines = 0

    for i, folder in enumerate(page_folders):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing page {i+1}/{len(page_folders)}...")

        extracted = extract_lines_from_page(folder, OUTPUT_DIR, i)
        total_lines += extracted

    print()
    print("="*60)
    print(f"Extraction complete!")
    print(f"Total line images extracted: {total_lines}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("To use in training, add to your training command:")
    print(f'  "training_data_lines/Arshasb_extracted/*.png"')


if __name__ == "__main__":
    main()
