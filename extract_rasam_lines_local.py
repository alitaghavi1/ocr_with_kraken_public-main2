"""
Extract line images from RASAM dataset using locally downloaded images.
Run download_rasam_images.py first!
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"
IMAGES_DIR = BASE_DIR / "training_data_lines" / "rasam_images"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasam_lines"


def parse_page_xml(xml_path):
    """Parse PAGE XML and extract text lines with coordinates."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

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

                text_elem = textline.find('page:TextEquiv/page:Unicode', ns)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text:
                    continue

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Line Extractor (Local)")
    print("="*50)

    xml_files = list(REPO_DIR.glob("page/**/*.xml"))
    print(f"Found {len(xml_files)} PAGE XML files")

    # Check images
    available_images = sum(1 for x in xml_files if (IMAGES_DIR / f"{x.stem}.jpg").exists())
    print(f"Available images: {available_images}/{len(xml_files)}\n")

    if available_images == 0:
        print("No images found! Run download_rasam_images.py first.")
        return

    total_lines = 0
    total_pages = 0

    for i, xml_path in enumerate(xml_files):
        image_id = xml_path.stem
        img_path = IMAGES_DIR / f"{image_id}.jpg"

        if not img_path.exists():
            continue

        # Parse XML
        lines = parse_page_xml(xml_path)
        if not lines:
            continue

        # Open image
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
        except Exception as e:
            print(f"  Error opening {image_id}: {e}")
            continue

        # Extract lines
        lines_saved = 0
        for line_data in lines:
            bbox = line_data['bbox']
            text = line_data['text']

            padding = 5
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(img.width, bbox[2] + padding)
            y2 = min(img.height, bbox[3] + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                line_img = img.crop((x1, y1, x2, y2))

                if line_img.width < 20 or line_img.height < 10:
                    continue

                out_png = OUTPUT_DIR / f"rasam_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"rasam_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(text, encoding='utf-8')

                total_lines += 1
                lines_saved += 1

            except:
                pass

        total_pages += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(xml_files)} pages, {total_lines} lines")

    print(f"\n{'='*50}")
    print(f"RASAM extraction complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
