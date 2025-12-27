"""
Extract line images from RASAM dataset with proper coordinate scaling.
Reads reference image dimensions from PAGE XML and scales coordinates
to match actual downloaded IIIF image dimensions.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import re

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"
IMAGES_DIR = BASE_DIR / "training_data_lines" / "rasam_images"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasam_lines_v2"


def parse_page_xml_with_dimensions(xml_path):
    """Parse PAGE XML, extract lines and reference image dimensions."""
    lines = []
    ref_width = None
    ref_height = None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try different namespace versions
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
        ]

        # Also try without namespace (some XMLs might not use namespace prefix)
        for ns in namespaces:
            # Get Page element for image dimensions
            page_elem = root.find('.//page:Page', ns)
            if page_elem is not None:
                ref_width = page_elem.get('imageWidth')
                ref_height = page_elem.get('imageHeight')
                if ref_width:
                    ref_width = int(ref_width)
                if ref_height:
                    ref_height = int(ref_height)

            for textline in root.findall('.//page:TextLine', ns):
                coords_elem = textline.find('page:Coords', ns)
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points', '')
                if not coords_str:
                    continue

                text_elem = textline.find('page:TextEquiv/page:Unicode', ns)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text:
                    continue

                # Parse polygon coordinates
                points = []
                for point in coords_str.split():
                    try:
                        x, y = map(float, point.split(','))
                        points.append((x, y))
                    except:
                        continue

                if len(points) >= 3:  # Need at least 3 points for a polygon
                    lines.append({'points': points, 'text': text})

            if lines:
                break

        # Try without namespace if nothing found
        if not lines:
            page_elem = root.find('.//Page')
            if page_elem is not None:
                ref_width = page_elem.get('imageWidth')
                ref_height = page_elem.get('imageHeight')
                if ref_width:
                    ref_width = int(ref_width)
                if ref_height:
                    ref_height = int(ref_height)

            for textline in root.findall('.//TextLine'):
                coords_elem = textline.find('Coords')
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points', '')
                if not coords_str:
                    continue

                text_elem = textline.find('TextEquiv/Unicode')
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text:
                    continue

                points = []
                for point in coords_str.split():
                    try:
                        x, y = map(float, point.split(','))
                        points.append((x, y))
                    except:
                        continue

                if len(points) >= 3:
                    lines.append({'points': points, 'text': text})

    except Exception as e:
        print(f"    XML parse error: {e}")

    return lines, ref_width, ref_height


def scale_and_crop_line(img, points, scale_x, scale_y, padding=10):
    """Scale coordinates and crop line from image."""
    # Scale all points
    scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in points]

    # Get bounding box
    xs = [p[0] for p in scaled_points]
    ys = [p[1] for p in scaled_points]

    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(img.width, int(max(xs)) + padding)
    y2 = min(img.height, int(max(ys)) + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Line Extractor v2 (with coordinate scaling)")
    print("=" * 60)

    xml_files = sorted(REPO_DIR.glob("page/**/*.xml"))
    print(f"Found {len(xml_files)} PAGE XML files")

    # Check images
    available_images = sum(1 for x in xml_files if (IMAGES_DIR / f"{x.stem}.jpg").exists())
    print(f"Available images: {available_images}/{len(xml_files)}\n")

    if available_images == 0:
        print("No images found! Run download_rasam_images.py first.")
        return

    total_lines = 0
    total_pages = 0
    skipped_scaling = 0
    skipped_size = 0
    skipped_text = 0

    for i, xml_path in enumerate(xml_files):
        image_id = xml_path.stem
        img_path = IMAGES_DIR / f"{image_id}.jpg"

        if not img_path.exists():
            continue

        # Parse XML with dimensions
        lines, ref_width, ref_height = parse_page_xml_with_dimensions(xml_path)

        if not lines:
            continue

        # Open image
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"  Error opening {image_id}: {e}")
            continue

        # Calculate scale factors
        if ref_width and ref_height:
            scale_x = img.width / ref_width
            scale_y = img.height / ref_height
        else:
            # If no reference dimensions, skip this file
            print(f"  Warning: No reference dimensions in {image_id}, skipping")
            skipped_scaling += 1
            continue

        # Convert to grayscale
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        # Extract lines
        lines_saved = 0
        for line_data in lines:
            points = line_data['points']
            text = line_data['text']

            # Skip very short text (likely noise)
            if len(text) < 2:
                skipped_text += 1
                continue

            try:
                line_img = scale_and_crop_line(img_gray, points, scale_x, scale_y, padding=8)

                if line_img is None:
                    continue

                # Skip very small images
                if line_img.width < 30 or line_img.height < 15:
                    skipped_size += 1
                    continue

                # Skip images that are too tall (likely multiple lines or errors)
                if line_img.height > line_img.width * 0.8:
                    skipped_size += 1
                    continue

                out_png = OUTPUT_DIR / f"rasam_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"rasam_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(text, encoding='utf-8')

                total_lines += 1
                lines_saved += 1

            except Exception as e:
                pass

        total_pages += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(xml_files)} pages, {total_lines} lines")

    print(f"\n{'=' * 60}")
    print(f"RASAM extraction v2 complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Skipped (no scaling info): {skipped_scaling}")
    print(f"  Skipped (too small/tall): {skipped_size}")
    print(f"  Skipped (short text): {skipped_text}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
