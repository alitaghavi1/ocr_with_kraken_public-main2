"""
Extract line images from RASM 2019 PAGE XML format for Kraken training.
Converts TIFF + PAGE XML to PNG line images + .gt.txt files.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import re

# Paths
BASE_DIR = Path(__file__).parent
RASM_DIRS = [
    BASE_DIR / "training_data_lines" / "RASM2019_part_1",
    BASE_DIR / "training_data_lines" / "RASM2019_part_2",
]
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasm_lines"

# PAGE XML namespace
NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}


def parse_coords(coords_str):
    """Parse polygon coordinates and return bounding box (x1, y1, x2, y2)."""
    points = []
    for point in coords_str.split():
        x, y = map(int, point.split(','))
        points.append((x, y))

    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def extract_lines_from_page(xml_path, tiff_path):
    """Extract all text lines from a PAGE XML file."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find all TextLine elements
        for textline in root.findall('.//page:TextLine', NS):
            # Get coordinates
            coords_elem = textline.find('page:Coords', NS)
            if coords_elem is None:
                continue

            coords_str = coords_elem.get('points', '')
            if not coords_str:
                continue

            bbox = parse_coords(coords_str)
            if bbox is None:
                continue

            # Get text content
            text_equiv = textline.find('page:TextEquiv/page:Unicode', NS)
            if text_equiv is None or text_equiv.text is None:
                continue

            text = text_equiv.text.strip()
            if not text:
                continue

            lines.append({
                'bbox': bbox,
                'text': text
            })

    except Exception as e:
        print(f"  Error parsing {xml_path.name}: {e}")

    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_pages = 0

    for rasm_dir in RASM_DIRS:
        if not rasm_dir.exists():
            print(f"Directory not found: {rasm_dir}")
            continue

        print(f"\nProcessing: {rasm_dir.name}")

        # Find all XML files
        xml_files = list(rasm_dir.glob("*.xml"))
        print(f"  Found {len(xml_files)} XML files")

        for xml_path in xml_files:
            # Find corresponding TIFF
            tiff_path = xml_path.with_suffix('.tif')
            if not tiff_path.exists():
                print(f"  TIFF not found for {xml_path.name}")
                continue

            # Extract lines from XML
            lines = extract_lines_from_page(xml_path, tiff_path)
            if not lines:
                continue

            # Open TIFF image
            try:
                img = Image.open(tiff_path)
                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')
            except Exception as e:
                print(f"  Error opening {tiff_path.name}: {e}")
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

                # Crop line
                try:
                    line_img = img.crop((x1, y1, x2, y2))

                    # Skip very small images
                    if line_img.width < 20 or line_img.height < 10:
                        continue

                    # Save
                    out_png = OUTPUT_DIR / f"rasm_{total_lines:05d}.png"
                    out_gt = OUTPUT_DIR / f"rasm_{total_lines:05d}.gt.txt"

                    line_img.save(out_png)
                    out_gt.write_text(text, encoding='utf-8')

                    total_lines += 1

                except Exception as e:
                    print(f"  Error cropping line: {e}")
                    continue

            total_pages += 1

            if total_pages % 10 == 0:
                print(f"  Processed {total_pages} pages, {total_lines} lines...")

    print(f"\n{'='*50}")
    print(f"RASM extraction complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
