"""
Extract line images from RASAM dataset using BASELINE for precise cropping.
The Baseline element gives exact text position - we crop a fixed height above/below it.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"
IMAGES_DIR = BASE_DIR / "training_data_lines" / "rasam_images"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasam_lines_v3"

# Height above and below baseline (in reference image pixels)
# Arabic script in RASAM: tighter cropping needed
ASCENDER_HEIGHT = 18  # pixels above baseline
DESCENDER_HEIGHT = 12  # pixels below baseline


def parse_page_xml_with_baselines(xml_path):
    """Parse PAGE XML, extract lines using Baseline for precise positioning."""
    lines = []
    ref_width = None
    ref_height = None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
        ]

        for ns in namespaces:
            page_elem = root.find('.//page:Page', ns)
            if page_elem is not None:
                ref_width = page_elem.get('imageWidth')
                ref_height = page_elem.get('imageHeight')
                if ref_width:
                    ref_width = int(ref_width)
                if ref_height:
                    ref_height = int(ref_height)

            for textline in root.findall('.//page:TextLine', ns):
                # Get baseline - this is the key for precise positioning
                baseline_elem = textline.find('page:Baseline', ns)
                coords_elem = textline.find('page:Coords', ns)

                if baseline_elem is None and coords_elem is None:
                    continue

                text_elem = textline.find('page:TextEquiv/page:Unicode', ns)
                if text_elem is None or text_elem.text is None:
                    continue

                text = text_elem.text.strip()
                if not text or len(text) < 2:
                    continue

                # Parse baseline points
                baseline_points = []
                if baseline_elem is not None:
                    baseline_str = baseline_elem.get('points', '')
                    for point in baseline_str.split():
                        try:
                            x, y = map(float, point.split(','))
                            baseline_points.append((x, y))
                        except:
                            continue

                # Parse coords for x-range (horizontal extent)
                coord_points = []
                if coords_elem is not None:
                    coords_str = coords_elem.get('points', '')
                    for point in coords_str.split():
                        try:
                            x, y = map(float, point.split(','))
                            coord_points.append((x, y))
                        except:
                            continue

                if baseline_points:
                    # Use baseline for y positioning
                    baseline_y = sum(p[1] for p in baseline_points) / len(baseline_points)
                    x_min = min(p[0] for p in baseline_points)
                    x_max = max(p[0] for p in baseline_points)

                    # Extend x-range from coords if available
                    if coord_points:
                        x_min = min(x_min, min(p[0] for p in coord_points))
                        x_max = max(x_max, max(p[0] for p in coord_points))

                    lines.append({
                        'x_min': x_min,
                        'x_max': x_max,
                        'baseline_y': baseline_y,
                        'text': text
                    })
                elif coord_points:
                    # Fallback: use middle of coord polygon
                    xs = [p[0] for p in coord_points]
                    ys = [p[1] for p in coord_points]
                    # Estimate baseline as 60% down from top (typical for Arabic)
                    y_min, y_max = min(ys), max(ys)
                    baseline_y = y_min + (y_max - y_min) * 0.6

                    lines.append({
                        'x_min': min(xs),
                        'x_max': max(xs),
                        'baseline_y': baseline_y,
                        'text': text
                    })

            if lines:
                break

    except ET.ParseError as e:
        pass  # Skip malformed XML
    except Exception as e:
        print(f"    Error: {e}")

    return lines, ref_width, ref_height


def crop_line_from_baseline(img, x_min, x_max, baseline_y, scale_x, scale_y,
                            ascender_px=ASCENDER_HEIGHT, descender_px=DESCENDER_HEIGHT):
    """Crop line using baseline position with fixed heights above/below."""
    # Scale coordinates
    x1 = int(x_min * scale_x)
    x2 = int(x_max * scale_x)
    baseline_scaled = baseline_y * scale_y

    # Scale the ascender/descender heights too
    ascender_scaled = ascender_px * scale_y
    descender_scaled = descender_px * scale_y

    # Calculate crop bounds
    y1 = int(baseline_scaled - ascender_scaled)
    y2 = int(baseline_scaled + descender_scaled)

    # Add small horizontal padding
    padding_x = 5
    x1 = max(0, x1 - padding_x)
    x2 = min(img.width, x2 + padding_x)
    y1 = max(0, y1)
    y2 = min(img.height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Line Extractor v3 (Baseline-based cropping)")
    print("=" * 60)
    print(f"Ascender height: {ASCENDER_HEIGHT}px, Descender: {DESCENDER_HEIGHT}px")
    print()

    xml_files = sorted(REPO_DIR.glob("page/**/*.xml"))
    print(f"Found {len(xml_files)} PAGE XML files")

    available_images = sum(1 for x in xml_files if (IMAGES_DIR / f"{x.stem}.jpg").exists())
    print(f"Available images: {available_images}/{len(xml_files)}\n")

    if available_images == 0:
        print("No images found! Run download_rasam_images.py first.")
        return

    total_lines = 0
    total_pages = 0
    skipped_size = 0
    skipped_nodim = 0

    for i, xml_path in enumerate(xml_files):
        image_id = xml_path.stem
        img_path = IMAGES_DIR / f"{image_id}.jpg"

        if not img_path.exists():
            continue

        lines, ref_width, ref_height = parse_page_xml_with_baselines(xml_path)

        if not lines:
            continue

        try:
            img = Image.open(img_path)
        except Exception as e:
            continue

        if not ref_width or not ref_height:
            skipped_nodim += 1
            continue

        scale_x = img.width / ref_width
        scale_y = img.height / ref_height

        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        lines_saved = 0
        for line_data in lines:
            try:
                line_img = crop_line_from_baseline(
                    img_gray,
                    line_data['x_min'],
                    line_data['x_max'],
                    line_data['baseline_y'],
                    scale_x, scale_y
                )

                if line_img is None:
                    continue

                # Quality checks
                if line_img.width < 50 or line_img.height < 20:
                    skipped_size += 1
                    continue

                # Skip if aspect ratio is wrong (too square or too tall)
                aspect = line_img.width / line_img.height
                if aspect < 3:  # Lines should be much wider than tall
                    skipped_size += 1
                    continue

                out_png = OUTPUT_DIR / f"rasam_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"rasam_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(line_data['text'], encoding='utf-8')

                total_lines += 1
                lines_saved += 1

            except Exception as e:
                pass

        total_pages += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(xml_files)} pages, {total_lines} lines")

    print(f"\n{'=' * 60}")
    print(f"RASAM extraction v3 complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Skipped (no dimensions): {skipped_nodim}")
    print(f"  Skipped (size/aspect): {skipped_size}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
