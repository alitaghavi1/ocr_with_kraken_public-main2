"""
Download and extract line images from OpenITI Arabic MS Data repository.
https://github.com/OpenITI/arabic_ms_data

Datasets available:
- hafiz_divan (Persian poetry)
- sadi_gulistan (Persian prose)
- firuzabadi_al_qamus_al_muhit (Arabic dictionary)
- taftazani_sharh_al_aqaid_al_nasafiya (Arabic theology)
- al_jazuli_dalail_al_khayrat (Arabic devotional)
- And more...

Uses ALTO XML format for ground truth.
"""

import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import io
import json
import ssl
import time

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "openiti_lines"
TEMP_DIR = BASE_DIR / "training_data_lines" / "openiti_temp"

# GitHub API and raw content URLs
GITHUB_API = "https://api.github.com/repos/OpenITI/arabic_ms_data/contents"
GITHUB_RAW = "https://raw.githubusercontent.com/OpenITI/arabic_ms_data/main"

# SSL context
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# Datasets to download (Persian and Arabic manuscripts)
DATASETS = [
    "hafiz_divan",           # Persian poetry - Hafiz
    "sadi_gulistan",         # Persian prose - Sa'di
    "firuzabadi_al_qamus_al_muhit",  # Arabic dictionary
    "taftazani_sharh_al_aqaid_al_nasafiya",  # Arabic theology
    "al_jazuli_dalail_al_khayrat",  # Arabic devotional
]


def fetch_json(url):
    """Fetch JSON from GitHub API."""
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/vnd.github.v3+json'
        })
        with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"    Error fetching {url}: {e}")
        return None


def fetch_file(url, dest):
    """Download file from URL."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


def parse_alto_xml(xml_path):
    """Parse ALTO XML and extract text lines with coordinates."""
    lines = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # ALTO namespace
        ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}

        # Try with namespace first
        textlines = root.findall('.//alto:TextLine', ns)

        # If not found, try without namespace
        if not textlines:
            # Remove namespace from all elements
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}')[1]
            textlines = root.findall('.//TextLine')

        for textline in textlines:
            # Get coordinates
            hpos = textline.get('HPOS')
            vpos = textline.get('VPOS')
            width = textline.get('WIDTH')
            height = textline.get('HEIGHT')

            if not all([hpos, vpos, width, height]):
                continue

            # Get text content from String elements
            text_parts = []
            for string in textline.findall('.//String', ns) or textline.findall('.//String'):
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
        print(f"    XML parse error: {e}")

    return lines


def get_manuscript_folders(dataset_name):
    """Get list of manuscript folders in a dataset."""
    url = f"{GITHUB_API}/{dataset_name}"
    data = fetch_json(url)

    if not data:
        return []

    folders = []
    for item in data:
        if item['type'] == 'dir' and not item['name'].startswith('.'):
            folders.append(item['name'])

    return folders


def get_page_files(dataset_name, manuscript_name):
    """Get list of PNG/XML file pairs in a manuscript folder."""
    url = f"{GITHUB_API}/{dataset_name}/{manuscript_name}"
    data = fetch_json(url)

    if not data:
        return []

    # Find PNG files and their corresponding XML
    png_files = {}
    xml_files = {}

    for item in data:
        if item['type'] == 'file':
            name = item['name']
            if name.endswith('.png'):
                base = name[:-4]
                png_files[base] = item['download_url']
            elif name.endswith('.xml'):
                base = name[:-4]
                xml_files[base] = item['download_url']

    # Return paired files
    pairs = []
    for base in png_files:
        if base in xml_files:
            pairs.append({
                'base': base,
                'png_url': png_files[base],
                'xml_url': xml_files[base]
            })

    return pairs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("OpenITI Arabic MS Data Downloader")
    print("=" * 60)
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"Output: {OUTPUT_DIR}\n")

    total_lines = 0
    total_pages = 0

    for dataset in DATASETS:
        print(f"\n[Dataset: {dataset}]")

        manuscripts = get_manuscript_folders(dataset)
        print(f"  Found {len(manuscripts)} manuscripts")

        for ms_idx, manuscript in enumerate(manuscripts):
            print(f"  [{ms_idx+1}/{len(manuscripts)}] {manuscript}")

            page_pairs = get_page_files(dataset, manuscript)
            print(f"    Pages with XML: {len(page_pairs)}")

            for pair in page_pairs:
                # Download PNG
                png_temp = TEMP_DIR / f"{pair['base']}.png"
                xml_temp = TEMP_DIR / f"{pair['base']}.xml"

                if not png_temp.exists():
                    if not fetch_file(pair['png_url'], png_temp):
                        continue

                if not xml_temp.exists():
                    if not fetch_file(pair['xml_url'], xml_temp):
                        continue

                # Parse XML
                lines = parse_alto_xml(xml_temp)

                if not lines:
                    continue

                # Open image
                try:
                    img = Image.open(png_temp)
                    if img.mode != 'L':
                        img = img.convert('L')
                except Exception as e:
                    print(f"    Error opening image: {e}")
                    continue

                # Extract lines
                for line_data in lines:
                    x, y, w, h = line_data['x'], line_data['y'], line_data['w'], line_data['h']
                    text = line_data['text']

                    # Add padding
                    padding = 5
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(img.width, x + w + padding)
                    y2 = min(img.height, y + h + padding)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    try:
                        line_img = img.crop((x1, y1, x2, y2))

                        # Skip very small images
                        if line_img.width < 50 or line_img.height < 15:
                            continue

                        # Save
                        out_png = OUTPUT_DIR / f"openiti_{total_lines:05d}.png"
                        out_gt = OUTPUT_DIR / f"openiti_{total_lines:05d}.gt.txt"

                        line_img.save(out_png)
                        out_gt.write_text(text, encoding='utf-8')

                        total_lines += 1

                    except Exception as e:
                        pass

                total_pages += 1

                # Rate limiting
                time.sleep(0.1)

            # Progress
            if total_lines > 0 and total_lines % 100 == 0:
                print(f"    Total lines so far: {total_lines}")

    print(f"\n{'=' * 60}")
    print(f"OpenITI extraction complete!")
    print(f"  Pages processed: {total_pages}")
    print(f"  Lines extracted: {total_lines}")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
