"""
Re-extract Arabic OpenITI lines with tighter cropping.
Reduces vertical margins to avoid capturing partial characters from adjacent lines.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "training_data_lines" / "openiti_temp"
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "openiti_lines_arabic"

# Arabic datasets from OpenITI
ARABIC_DATASETS = [
    "firuzabadi_al_qamus_al_muhit",
    "taftazani_sharh_al_aqaid_al_nasafiya",
    "al_jazuli_dalail_al_khayrat",
]

# Very aggressive vertical cropping - crop well inside the bounding box
VERTICAL_PADDING = -12  # Crop 12px into the box from top/bottom
HORIZONTAL_PADDING = 2


def parse_alto_xml(xml_path):
    """Parse ALTO XML and extract text lines with coordinates."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Handle namespace
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}')[1]

        for textline in root.findall('.//TextLine'):
            hpos = textline.get('HPOS')
            vpos = textline.get('VPOS')
            width = textline.get('WIDTH')
            height = textline.get('HEIGHT')

            if not all([hpos, vpos, width, height]):
                continue

            # Get text content
            text_parts = []
            for string in textline.findall('.//String'):
                content = string.get('CONTENT', '')
                if content:
                    text_parts.append(content)

            text = ' '.join(text_parts).strip()

            if text and len(text) >= 2:
                lines.append({
                    'x': int(float(hpos)),
                    'y': int(float(vpos)),
                    'w': int(float(width)),
                    'h': int(float(height)),
                    'text': text
                })

    except Exception as e:
        pass

    return lines


def main():
    print("Re-cropping Arabic OpenITI with tighter margins")
    print("=" * 60)
    print(f"Vertical padding: {VERTICAL_PADDING}px")
    print(f"Horizontal padding: {HORIZONTAL_PADDING}px")
    print()

    # Clear output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*.png"):
        f.unlink()
    for f in OUTPUT_DIR.glob("*.gt.txt"):
        f.unlink()

    # Find all XML files in temp that match Arabic datasets
    xml_files = []
    for xml_path in TEMP_DIR.glob("*.xml"):
        # Check if this file is from an Arabic dataset
        for dataset in ARABIC_DATASETS:
            if dataset in xml_path.stem.lower() or any(
                part in xml_path.stem.lower()
                for part in ['arabe', 'arabic', 'ara_', 'glaser', 'wetzstein', 'garrett', 'vollers', 'gabelentz']
            ):
                xml_files.append(xml_path)
                break

    # Also check by looking at paired PNG files
    all_xmls = list(TEMP_DIR.glob("*.xml"))
    print(f"Total XML files in temp: {len(all_xmls)}")

    total_lines = 0
    processed_files = 0

    for xml_path in all_xmls:
        png_path = xml_path.with_suffix('.png')

        if not png_path.exists():
            continue

        # Parse XML
        lines = parse_alto_xml(xml_path)

        if not lines:
            continue

        # Only include Arabic files - whitelist approach
        fname_lower = xml_path.stem.lower()

        # Must contain Arabic indicators
        arabic_indicators = ['arabe', 'arabic', 'ara_', 'glaser', 'wetzstein',
                            'vollers', 'gabelentz', 'garrett', 'jazuli',
                            'qamus', 'taftazani', 'nasafi', 'dalail']

        # Must NOT contain Persian indicators
        persian_indicators = ['persan', 'persian', 'pers_', 'hafiz', 'gulistan',
                             'divan', 'sadi']

        is_arabic = any(p in fname_lower for p in arabic_indicators)
        is_persian = any(p in fname_lower for p in persian_indicators)

        if not is_arabic or is_persian:
            continue

        # Open image
        try:
            img = Image.open(png_path)
            if img.mode != 'L':
                img = img.convert('L')
        except Exception as e:
            continue

        # Extract lines with tight cropping
        for line_data in lines:
            x, y, w, h = line_data['x'], line_data['y'], line_data['w'], line_data['h']
            text = line_data['text']

            # Fixed height approach - estimate proper line height from width
            # Arabic text lines typically have aspect ratio of 15:1 to 25:1
            estimated_height = max(40, w // 18)  # Reasonable line height

            # Center vertically in the bounding box
            center_y = y + h // 2

            x1 = max(0, x - HORIZONTAL_PADDING)
            y1 = max(0, center_y - estimated_height // 2)
            x2 = min(img.width, x + w + HORIZONTAL_PADDING)
            y2 = min(img.height, center_y + estimated_height // 2)

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                line_img = img.crop((x1, y1, x2, y2))

                # Skip very small images
                if line_img.width < 50 or line_img.height < 15:
                    continue

                # Save
                out_png = OUTPUT_DIR / f"openiti_ar_{total_lines:05d}.png"
                out_gt = OUTPUT_DIR / f"openiti_ar_{total_lines:05d}.gt.txt"

                line_img.save(out_png)
                out_gt.write_text(text, encoding='utf-8')

                total_lines += 1

            except Exception:
                pass

        processed_files += 1

        if processed_files % 20 == 0:
            print(f"  Processed {processed_files} files, {total_lines} lines")

    print(f"\n{'=' * 60}")
    print(f"Re-cropping complete!")
    print(f"  Files processed: {processed_files}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
