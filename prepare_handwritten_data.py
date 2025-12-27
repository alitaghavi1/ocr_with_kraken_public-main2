"""
Prepare Handwritten Training Data for Kraken Fine-Tuning

This script helps you prepare training data from handwritten samples.

Supported input formats:
1. Page images that need to be segmented into lines
2. Pre-segmented line images
3. Labeled character/word images (from labeling tools)

Usage:
    python prepare_handwritten_data.py segment <page_images_folder> <output_folder>
    python prepare_handwritten_data.py from-lines <line_images_folder> <output_folder>
    python prepare_handwritten_data.py validate <training_folder>
    python prepare_handwritten_data.py stats <training_folder>
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "handwritten_training_data"

# Standard height for Kraken training
TARGET_HEIGHT = 48


def resize_image_to_height(image_path, target_height=TARGET_HEIGHT):
    """Resize image to standard height while preserving aspect ratio"""
    img = Image.open(image_path)
    width, height = img.size

    if height != target_height:
        ratio = target_height / height
        new_width = int(width * ratio)
        img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)

    return img


def segment_page_to_lines(page_path, output_dir, direction="horizontal-rl"):
    """
    Segment a page image into individual line images using Kraken.

    Args:
        page_path: Path to the page image
        output_dir: Directory to save line images
        direction: Text direction (horizontal-rl for RTL, horizontal-lr for LTR)
    """
    page_path = Path(page_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting: {page_path.name}")

    # Use kraken to segment
    json_output = output_dir / f"{page_path.stem}_seg.json"

    cmd = [
        "kraken", "-i", str(page_path), str(json_output),
        "binarize", "segment", "-d", direction, "-bl"
    ]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"  Segmentation failed: {result.stderr}")
        return []

    # Extract lines from the segmentation
    try:
        import json
        with open(json_output, 'r', encoding='utf-8') as f:
            seg_data = json.load(f)

        # Load the original image
        img = Image.open(page_path)

        lines = []
        for i, line in enumerate(seg_data.get('lines', [])):
            # Get bounding box
            if 'boundary' in line:
                coords = line['boundary']
                min_x = min(p[0] for p in coords)
                max_x = max(p[0] for p in coords)
                min_y = min(p[1] for p in coords)
                max_y = max(p[1] for p in coords)
            elif 'bbox' in line:
                min_x, min_y, max_x, max_y = line['bbox']
            else:
                continue

            # Crop line
            line_img = img.crop((min_x, min_y, max_x, max_y))

            # Save line image
            line_name = f"{page_path.stem}_line{i:03d}.png"
            line_path = output_dir / line_name
            line_img.save(line_path)

            # Create empty ground truth file
            gt_path = line_path.with_suffix('.gt.txt')
            gt_path.write_text("", encoding='utf-8')

            lines.append(line_path)
            print(f"  Extracted: {line_name}")

        return lines

    except Exception as e:
        print(f"  Error extracting lines: {e}")
        return []


def prepare_line_images(input_dir, output_dir):
    """
    Prepare pre-segmented line images for training.
    Resizes images and creates empty ground truth files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = [f for f in input_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images in {input_dir}")

    for i, img_path in enumerate(sorted(images)):
        # Resize to standard height
        img = resize_image_to_height(img_path)

        # Save with standardized name
        out_name = f"hw_{i:06d}.png"
        out_path = output_dir / out_name
        img.save(out_path)

        # Check for existing ground truth
        gt_source = img_path.with_suffix('.gt.txt')
        gt_dest = out_path.with_suffix('.gt.txt')

        if gt_source.exists():
            shutil.copy(gt_source, gt_dest)
            print(f"  {img_path.name} -> {out_name} (with ground truth)")
        else:
            gt_dest.write_text("", encoding='utf-8')
            print(f"  {img_path.name} -> {out_name} (needs transcription)")

    print(f"\nPrepared {len(images)} images in {output_dir}")
    print("\nNext steps:")
    print("1. Open each .gt.txt file and add the correct transcription")
    print("2. Run: python prepare_handwritten_data.py validate", output_dir)


def validate_training_data(training_dir):
    """Validate that training data is properly formatted"""
    training_dir = Path(training_dir)

    if not training_dir.exists():
        print(f"ERROR: Directory not found: {training_dir}")
        return False

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    images = [f for f in training_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"Validating training data in: {training_dir}")
    print(f"Found {len(images)} images")
    print("-" * 50)

    valid = 0
    empty_gt = 0
    missing_gt = 0
    errors = []

    for img_path in sorted(images):
        gt_path = img_path.with_suffix('.gt.txt')

        if not gt_path.exists():
            missing_gt += 1
            errors.append(f"Missing: {gt_path.name}")
        else:
            content = gt_path.read_text(encoding='utf-8').strip()
            if not content:
                empty_gt += 1
                errors.append(f"Empty: {gt_path.name}")
            else:
                valid += 1

    print(f"Valid pairs:     {valid}")
    print(f"Empty GT files:  {empty_gt}")
    print(f"Missing GT:      {missing_gt}")
    print("-" * 50)

    if errors and len(errors) <= 20:
        print("\nIssues:")
        for e in errors:
            print(f"  - {e}")
    elif errors:
        print(f"\n{len(errors)} issues found (showing first 20):")
        for e in errors[:20]:
            print(f"  - {e}")

    if valid == 0:
        print("\nERROR: No valid training pairs found!")
        print("Each image needs a .gt.txt file with the transcription")
        return False

    if valid < 100:
        print(f"\nWARNING: Only {valid} samples. Consider adding more data.")
        print("Recommended: 500+ samples for fine-tuning, 1000+ for training from scratch")

    print(f"\nReady for training with {valid} samples")
    return True


def show_stats(training_dir):
    """Show statistics about the training data"""
    training_dir = Path(training_dir)

    if not training_dir.exists():
        print(f"ERROR: Directory not found: {training_dir}")
        return

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    images = [f for f in training_dir.iterdir()
              if f.suffix.lower() in image_extensions]

    print(f"Training Data Statistics: {training_dir}")
    print("=" * 50)

    total_chars = 0
    char_counts = {}
    word_counts = {}
    line_lengths = []

    for img_path in images:
        gt_path = img_path.with_suffix('.gt.txt')
        if gt_path.exists():
            text = gt_path.read_text(encoding='utf-8').strip()
            if text:
                total_chars += len(text)
                line_lengths.append(len(text))

                # Count characters
                for char in text:
                    char_counts[char] = char_counts.get(char, 0) + 1

                # Count words
                words = text.split()
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1

    valid_samples = len(line_lengths)

    print(f"Total images:      {len(images)}")
    print(f"Valid samples:     {valid_samples}")
    print(f"Total characters:  {total_chars}")
    print(f"Unique characters: {len(char_counts)}")
    print(f"Unique words:      {len(word_counts)}")

    if line_lengths:
        avg_len = sum(line_lengths) / len(line_lengths)
        print(f"Avg line length:   {avg_len:.1f} chars")
        print(f"Min line length:   {min(line_lengths)} chars")
        print(f"Max line length:   {max(line_lengths)} chars")

    if char_counts:
        print("\nTop 20 characters:")
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])[:20]
        for char, count in sorted_chars:
            display = char if char.strip() else repr(char)
            print(f"  '{display}': {count}")


def print_usage():
    print(__doc__)
    print("\nExample workflow:")
    print("  1. From page images:")
    print("     python prepare_handwritten_data.py segment pages/ output/")
    print("     # Then manually transcribe each .gt.txt file")
    print("")
    print("  2. From line images:")
    print("     python prepare_handwritten_data.py from-lines lines/ output/")
    print("     # Then manually transcribe each .gt.txt file")
    print("")
    print("  3. Validate data:")
    print("     python prepare_handwritten_data.py validate handwritten_training_data/")
    print("")
    print("  4. View statistics:")
    print("     python prepare_handwritten_data.py stats handwritten_training_data/")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "segment":
        if len(sys.argv) < 4:
            print("Usage: python prepare_handwritten_data.py segment <pages_folder> <output_folder>")
            sys.exit(1)
        pages_dir = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])

        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.pdf'}
        pages = [f for f in pages_dir.iterdir()
                 if f.suffix.lower() in image_extensions]

        print(f"Found {len(pages)} page images")
        for page in pages:
            segment_page_to_lines(page, output_dir)

    elif command == "from-lines":
        if len(sys.argv) < 4:
            print("Usage: python prepare_handwritten_data.py from-lines <lines_folder> <output_folder>")
            sys.exit(1)
        prepare_line_images(sys.argv[2], sys.argv[3])

    elif command == "validate":
        folder = sys.argv[2] if len(sys.argv) > 2 else "handwritten_training_data"
        validate_training_data(folder)

    elif command == "stats":
        folder = sys.argv[2] if len(sys.argv) > 2 else "handwritten_training_data"
        show_stats(folder)

    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
