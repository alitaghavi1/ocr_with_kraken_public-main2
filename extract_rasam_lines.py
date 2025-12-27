"""
Extract line images from RASAM dataset (cloned repo).
- Downloads images from BULAC IIIF server
- Parses PAGE XML for line coordinates and transcriptions
- Saves as PNG + .gt.txt for Kraken training
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import urllib.request
import io
import time

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasam_lines"

# IIIF server base URL
IIIF_BASE = "https://bina.bulac.fr/iiif/2"


def parse_page_xml(xml_path):
    """Parse PAGE XML and extract text lines with coordinates."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Try different namespace versions
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
        ]

        for ns in namespaces:
            for textline in root.findall('.//page:TextLine', ns):
                coords_elem = textline.find('page:Coords', ns)
                if coords_elem is None:
                    continue

                coords_str = coords_elem.get('points', '')
                if not coords_str:
                    continue

                # Get text
                text_elem = textline.find('page:TextEquiv/page:Unicode', ns)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text:
                    continue

                # Parse coordinates to bounding box
                points = []
                for point in coords_str.split():
                    try:
                        x, y = map(float, point.split(','))
                        points.append((int(x), int(y)))
                    except:
                        continue

                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    lines.append({'bbox': bbox, 'text': text})

            if lines:
                break

    except Exception as e:
        print(f"    XML parse error: {e}")

    return lines


def download_iiif_image(image_id, width=2000):
    """Download image from BULAC IIIF server."""
    url = f"{IIIF_BASE}/{image_id}/full/{width},/0/default.jpg"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as response:
            return Image.open(io.BytesIO(response.read()))
    except Exception as e:
        # Try smaller size
        try:
            url = f"{IIIF_BASE}/{image_id}/full/1000,/0/default.jpg"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                return Image.open(io.BytesIO(response.read()))
        except:
            return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Line Extractor")
    print("="*50)

    # Find all PAGE XML files
    xml_files = list(REPO_DIR.glob("page/**/*.xml"))
    print(f"Found {len(xml_files)} PAGE XML files\n")

    total_lines = 0
    total_pages = 0
    failed_downloads = 0

    for i, xml_path in enumerate(xml_files):
        # Get image ID from filename
        image_id = xml_path.stem

        print(f"[{i+1}/{len(xml_files)}] {image_id}")

        # Parse XML
        lines = parse_page_xml(xml_path)
        if not lines:
            print(f"    No lines found")
            continue

        print(f"    Found {len(lines)} lines")

        # Download image from IIIF
        print(f"    Downloading image...")
        img = download_iiif_image(image_id)

        if img is None:
            print(f"    Failed to download image")
            failed_downloads += 1
            continue

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Get scale factor (IIIF might return different size)
        # We assume the coordinates in XML match the full resolution

        # Extract each line
        lines_saved = 0
        for line_data in lines:
            bbox = line_data['bbox']
            text = line_data['text']

            # Add padding
            padding = 5
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(img.width, bbox[2] + padding)
            y2 = min(img.height, bbox[3] + padding)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            try:
                line_img = img.crop((x1, y1, x2, y2))

                # Skip very small images
                if line_img.width < 20 or line_img.height < 10:
                    continue

                # Save
                out_png = OUTPUT_DIR / f"rasam_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"rasam_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(text, encoding='utf-8')

                total_lines += 1
                lines_saved += 1

            except Exception as e:
                pass

        print(f"    Saved {lines_saved} lines")
        total_pages += 1

        # Be nice to the server
        time.sleep(0.3)

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"\n--- Progress: {i+1}/{len(xml_files)} pages, {total_lines} lines ---\n")

    print(f"\n{'='*50}")
    print(f"RASAM extraction complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Failed downloads: {failed_downloads}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
