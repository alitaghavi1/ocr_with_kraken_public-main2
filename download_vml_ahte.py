"""
Download VML-AHTE Dataset - Arabic Handwritten Text Line Extraction

Source: https://github.com/beratkurar/arabic_handwritten_textline_extraction_dataset
Paper: VML-AHTE: A database for Arabic handwritten text line extraction

Contains:
- 20 training pages + 10 test pages
- Line-level pixel labels and PAGE XML ground truth
- Native Arabic speaker annotations
- Crowded diacritics, touching and overlapping characters
"""

import os
import subprocess
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# Configuration
REPO_URL = "https://github.com/beratkurar/arabic_handwritten_textline_extraction_dataset.git"
CLONE_DIR = Path("training_data_lines/vml_ahte_raw")
OUTPUT_DIR = Path("training_data_lines/vml_ahte_lines")

# PAGE XML namespace
PAGE_NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}


def clone_repo():
    """Clone the VML-AHTE repository."""
    if CLONE_DIR.exists():
        print(f"Repository already exists at {CLONE_DIR}")
        return True

    print(f"Cloning VML-AHTE repository...")
    print(f"URL: {REPO_URL}")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
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


def parse_page_xml(xml_path):
    """Parse PAGE XML and extract text lines with coordinates."""
    lines = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try different namespace versions
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
        ]

        for ns in namespaces:
            text_lines = root.findall('.//page:TextLine', ns)
            if not text_lines:
                continue

            for textline in text_lines:
                # Get coordinates
                coords_elem = textline.find('page:Coords', ns)
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points', '')
                if not coords_str:
                    continue

                # Get text content
                text_elem = textline.find('page:TextEquiv/page:Unicode', ns)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text:
                    continue

                # Parse coordinates to bounding box
                points = []
                for point in coords_str.split():
                    if ',' in point:
                        x, y = map(int, point.split(','))
                        points.append((x, y))

                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    lines.append({'bbox': bbox, 'text': text})

            if lines:
                break

    except Exception as e:
        print(f"  Error parsing {xml_path}: {e}")

    return lines


def extract_lines():
    """Extract text lines from page images using PAGE XML coordinates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_lines = 0

    # Look for PAGE XML files
    xml_files = list(CLONE_DIR.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    for xml_path in xml_files:
        print(f"\nProcessing: {xml_path.name}")

        # Find corresponding image
        possible_images = [
            xml_path.with_suffix('.png'),
            xml_path.with_suffix('.jpg'),
            xml_path.with_suffix('.tif'),
            xml_path.with_suffix('.tiff'),
        ]

        img_path = None
        for p in possible_images:
            if p.exists():
                img_path = p
                break

        if img_path is None:
            # Try looking in different locations
            for ext in ['.png', '.jpg', '.tif', '.tiff']:
                matches = list(xml_path.parent.glob(f"{xml_path.stem}*{ext}"))
                if matches:
                    img_path = matches[0]
                    break

        if img_path is None:
            print(f"  No image found for {xml_path.name}")
            continue

        # Parse XML
        lines = parse_page_xml(xml_path)
        print(f"  Found {len(lines)} text lines")

        if not lines:
            continue

        # Load image
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue

        # Extract each line
        for line_data in lines:
            bbox = line_data['bbox']
            text = line_data['text']

            # Add padding
            padding = 5
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(img.width, bbox[2] + padding)
            y2 = min(img.height, bbox[3] + padding)

            # Check minimum size
            if (x2 - x1) < 30 or (y2 - y1) < 10:
                continue

            try:
                line_img = img.crop((x1, y1, x2, y2))

                out_png = OUTPUT_DIR / f"vml_ahte_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"vml_ahte_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(text, encoding='utf-8')
                total_lines += 1

            except Exception as e:
                print(f"  Error extracting line: {e}")

    return total_lines


def main():
    print("=" * 60)
    print("VML-AHTE Dataset Downloader")
    print("Arabic Handwritten Text Line Extraction Database")
    print("=" * 60)
    print()
    print("Source: GitHub (beratkurar/arabic_handwritten_textline_extraction_dataset)")
    print("Content: 30 pages with line-level annotations")
    print("         Native Arabic speaker ground truth")
    print()

    # Step 1: Clone repository
    if not clone_repo():
        print("\nFailed to clone repository.")
        return

    # Step 2: List contents
    print("\nExploring repository structure...")
    for item in sorted(CLONE_DIR.iterdir()):
        if item.is_dir():
            count = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"  {item.name}/: {count} files")
        else:
            print(f"  {item.name}")

    # Step 3: Extract lines
    print("\nExtracting text lines from pages...")
    total = extract_lines()

    print()
    print("=" * 60)
    print(f"Download complete!")
    print(f"  Total line images extracted: {total}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
