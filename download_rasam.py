"""
Download RASAM dataset (Maghrebi Arabic manuscripts) and convert to Kraken format.
- 11,290 transcribed lines from 450 manuscript images
- Images from BULAC Library IIIF server
- Ground truth in PAGE XML format
"""

import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import io
import time

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "rasam_lines"
REPO_DIR = BASE_DIR / "training_data_lines" / "rasam_repo"

# GitHub raw URLs
GITHUB_RAW = "https://raw.githubusercontent.com/calfa-co/rasam-dataset/main"

# PAGE XML namespace
NS = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}


def download_file(url, dest):
    """Download file from URL."""
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"    Error downloading: {e}")
        return False


def get_image_list():
    """Download and parse the list of images."""
    list_url = f"{GITHUB_RAW}/list-images.tsv"
    print(f"Downloading image list from: {list_url}")

    try:
        with urllib.request.urlopen(list_url) as response:
            content = response.read().decode('utf-8')
            lines = content.strip().split('\n')
            # Skip header
            images = []
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    images.append(parts[0])
            return images
    except Exception as e:
        print(f"Error: {e}")
        return []


def download_page_xml(image_id):
    """Download PAGE XML for an image."""
    # Try different possible paths
    possible_paths = [
        f"{GITHUB_RAW}/data/{image_id}.xml",
        f"{GITHUB_RAW}/gt/{image_id}.xml",
        f"{GITHUB_RAW}/pagexml/{image_id}.xml",
    ]

    for url in possible_paths:
        try:
            with urllib.request.urlopen(url) as response:
                return response.read().decode('utf-8')
        except:
            continue
    return None


def download_iiif_image(image_id, width=2000):
    """Download image from BULAC IIIF server."""
    url = f"https://bina.bulac.fr/iiif/2/{image_id}/full/{width},/0/default.jpg"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return Image.open(io.BytesIO(response.read()))
    except Exception as e:
        print(f"    Error downloading image: {e}")
        return None


def parse_page_xml(xml_content):
    """Parse PAGE XML and extract text lines with coordinates."""
    lines = []
    try:
        root = ET.fromstring(xml_content)

        # Try different namespace versions
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
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
        print(f"    XML parse error: {e}")

    return lines


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPO_DIR.mkdir(parents=True, exist_ok=True)

    print("RASAM Dataset Downloader")
    print("="*50)

    # First, clone/download the repository structure
    print("\nStep 1: Getting image list...")
    image_ids = get_image_list()

    if not image_ids:
        print("Could not get image list. Trying to clone repo...")
        print("\nPlease run:")
        print("  git clone https://github.com/calfa-co/rasam-dataset.git training_data_lines/rasam_repo")
        print("\nThen run this script again.")
        return

    print(f"Found {len(image_ids)} images")

    # For now, let's try a sample to understand the structure
    print("\nStep 2: Testing with first 3 images...")

    total_lines = 0

    for i, image_id in enumerate(image_ids[:3]):
        print(f"\n[{i+1}] Processing: {image_id}")

        # Download PAGE XML
        xml_content = download_page_xml(image_id)
        if xml_content:
            print(f"    Got PAGE XML")
            lines = parse_page_xml(xml_content)
            print(f"    Found {len(lines)} lines")

            if lines:
                # Download image
                print(f"    Downloading image from IIIF...")
                img = download_iiif_image(image_id)

                if img:
                    # Convert to grayscale
                    if img.mode != 'L':
                        img = img.convert('L')

                    # Extract lines
                    for line_data in lines:
                        bbox = line_data['bbox']
                        text = line_data['text']

                        # Add padding
                        padding = 5
                        x1 = max(0, bbox[0] - padding)
                        y1 = max(0, bbox[1] - padding)
                        x2 = min(img.width, bbox[2] + padding)
                        y2 = min(img.height, bbox[3] + padding)

                        try:
                            line_img = img.crop((x1, y1, x2, y2))

                            if line_img.width > 20 and line_img.height > 10:
                                out_png = OUTPUT_DIR / f"rasam_{total_lines:05d}.png"
                                out_gt = OUTPUT_DIR / f"rasam_{total_lines:05d}.gt.txt"

                                line_img.save(out_png)
                                out_gt.write_text(text, encoding='utf-8')
                                total_lines += 1
                        except:
                            pass

                    print(f"    Extracted {len(lines)} lines")
        else:
            print(f"    No PAGE XML found")

        time.sleep(0.5)  # Be nice to the server

    print(f"\n{'='*50}")
    print(f"Test complete! Extracted {total_lines} lines")
    print(f"Output: {OUTPUT_DIR}")

    if total_lines > 0:
        print(f"\nTo download full dataset, modify the script to process all {len(image_ids)} images.")
    else:
        print("\nNote: The dataset structure may require cloning the full repo.")
        print("Run: git clone https://github.com/calfa-co/rasam-dataset.git")


if __name__ == "__main__":
    main()
