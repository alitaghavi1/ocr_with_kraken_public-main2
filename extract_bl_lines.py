"""
Extract line images and transcriptions from British Library PAGE XML dataset.

This script:
1. Parses PAGE XML files to get line coordinates and transcriptions
2. Crops line images from the TIFF files
3. Saves as image + .gt.txt pairs for Kraken training

PAGE XML format:
- TextRegion contains TextLine elements
- Each TextLine has Coords (polygon) and TextEquiv (transcription)
"""

import os
import re
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import warnings

# Suppress PIL warnings for large images
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None  # Allow large TIFF files

# Configuration
SOURCE_DIR = "training_data_lines/british_library_arabic"
OUTPUT_DIR = "training_data_lines/bl_extracted_lines"
PADDING = 10  # Pixels to add around each line crop

# PAGE XML namespace
PAGE_NS = {
    'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15',
    'page2017': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15',
    'page2019': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15',
}


def parse_coords(coords_str):
    """Parse coordinate string like '100,200 150,200 150,250 100,250' to list of (x,y) tuples."""
    points = []
    for point in coords_str.strip().split():
        if ',' in point:
            x, y = point.split(',')
            points.append((int(x), int(y)))
    return points


def get_bounding_box(points):
    """Get bounding box from list of points."""
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def parse_page_xml(xml_path):
    """Parse PAGE XML and extract text lines with coordinates and transcriptions."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try to detect namespace
        ns = None
        for prefix, uri in PAGE_NS.items():
            if uri in root.tag or root.find(f'.//{{{uri}}}TextLine') is not None:
                ns = {'ns': uri}
                break

        # Try without namespace if not found
        if ns is None:
            # Check if namespace is in root tag
            match = re.match(r'\{(.+?)\}', root.tag)
            if match:
                ns = {'ns': match.group(1)}
            else:
                ns = {}

        # Find all TextLine elements
        if ns:
            text_lines = root.findall('.//ns:TextLine', ns)
            if not text_lines:
                # Try alternative paths
                text_lines = root.findall('.//{%s}TextLine' % ns.get('ns', ''))
        else:
            text_lines = root.findall('.//TextLine')

        # Also try without namespace prefix
        if not text_lines:
            for elem in root.iter():
                if elem.tag.endswith('TextLine'):
                    text_lines.append(elem)

        for line_elem in text_lines:
            line_data = {'coords': None, 'text': None, 'id': None}

            # Get line ID
            line_data['id'] = line_elem.get('id', '')

            # Get coordinates
            coords_elem = None
            for child in line_elem:
                if child.tag.endswith('Coords'):
                    coords_elem = child
                    break

            if coords_elem is not None:
                points_str = coords_elem.get('points', '')
                if points_str:
                    points = parse_coords(points_str)
                    line_data['coords'] = get_bounding_box(points)

            # Get transcription
            for child in line_elem:
                if child.tag.endswith('TextEquiv'):
                    for subchild in child:
                        if subchild.tag.endswith('Unicode'):
                            line_data['text'] = subchild.text
                            break
                    break

            # Only add if we have both coordinates and text
            if line_data['coords'] and line_data['text']:
                lines.append(line_data)

    except ET.ParseError as e:
        print(f"  XML parse error in {xml_path}: {e}")
    except Exception as e:
        print(f"  Error processing {xml_path}: {e}")

    return lines


def extract_lines_from_page(image_path, xml_path, output_dir, page_id):
    """Extract all lines from a page image using PAGE XML annotations."""
    lines = parse_page_xml(xml_path)

    if not lines:
        print(f"  No lines found in {xml_path}")
        return 0

    # Load image
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed (TIFF might be in different modes)
        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
    except Exception as e:
        print(f"  Error loading image {image_path}: {e}")
        return 0

    extracted = 0
    img_width, img_height = img.size

    for i, line in enumerate(lines):
        bbox = line['coords']
        text = line['text'].strip()

        # Skip empty or very short transcriptions
        if not text or len(text) < 2:
            continue

        # Apply padding and ensure within bounds
        left = max(0, bbox[0] - PADDING)
        top = max(0, bbox[1] - PADDING)
        right = min(img_width, bbox[2] + PADDING)
        bottom = min(img_height, bbox[3] + PADDING)

        # Skip very small crops
        if (right - left) < 50 or (bottom - top) < 10:
            continue

        # Crop line
        try:
            line_img = img.crop((left, top, right, bottom))

            # Convert to grayscale for consistency
            if line_img.mode != 'L':
                line_img = line_img.convert('L')

            # Save files
            line_id = line.get('id', f'line{i:03d}')
            out_name = f"bl_{page_id}_{line_id}"
            out_img_path = os.path.join(output_dir, f"{out_name}.png")
            out_gt_path = os.path.join(output_dir, f"{out_name}.gt.txt")

            line_img.save(out_img_path)
            with open(out_gt_path, 'w', encoding='utf-8') as f:
                f.write(text)

            extracted += 1

        except Exception as e:
            print(f"  Error extracting line {i}: {e}")
            continue

    return extracted


def find_image_for_xml(xml_path, source_dir):
    """Find the corresponding image file for an XML file."""
    xml_name = Path(xml_path).stem

    # Common patterns: file.xml -> file.tif, file.tiff, file.png, file.jpg
    extensions = ['.tif', '.tiff', '.TIF', '.TIFF', '.png', '.PNG', '.jpg', '.JPG', '.jpeg']

    for ext in extensions:
        # Same directory
        img_path = Path(xml_path).with_suffix(ext)
        if img_path.exists():
            return str(img_path)

        # Search in source directory
        pattern = os.path.join(source_dir, '**', xml_name + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    return None


def main():
    print("=" * 60)
    print("Extracting lines from British Library PAGE XML dataset")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all XML files
    xml_files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.xml'), recursive=True)

    if not xml_files:
        print(f"No XML files found in {SOURCE_DIR}")
        print()
        print("Please first download the dataset:")
        print("  python download_bl_dataset.py")
        print()
        print("Or manually download from:")
        print("  https://zenodo.org/record/3271395")
        return

    print(f"Found {len(xml_files)} XML files")
    print(f"Output: {OUTPUT_DIR}")
    print()

    total_lines = 0
    processed_pages = 0

    for i, xml_path in enumerate(xml_files):
        xml_name = Path(xml_path).stem

        # Find corresponding image
        image_path = find_image_for_xml(xml_path, SOURCE_DIR)

        if not image_path:
            print(f"[{i+1}/{len(xml_files)}] {xml_name}: No image found, skipping")
            continue

        print(f"[{i+1}/{len(xml_files)}] Processing {xml_name}...")

        # Generate unique page ID
        page_id = f"p{i+1:03d}"

        # Extract lines
        extracted = extract_lines_from_page(image_path, xml_path, OUTPUT_DIR, page_id)

        if extracted > 0:
            print(f"  Extracted {extracted} lines")
            total_lines += extracted
            processed_pages += 1
        else:
            print(f"  No lines extracted")

    print()
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"  Pages processed: {processed_pages}")
    print(f"  Total lines extracted: {total_lines}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    print("To use in training:")
    print(f'  "training_data_lines/bl_extracted_lines/*.png"')


if __name__ == "__main__":
    main()
