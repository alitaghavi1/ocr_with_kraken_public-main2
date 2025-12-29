"""
Create synthetic line images by combining 3-4 word images horizontally.

This takes individual word images and concatenates them to create
realistic line training data for Kraken OCR.
"""

import random
import shutil
from pathlib import Path
from PIL import Image
import sys

# Paths
BASE_DIR = Path(__file__).parent
WORD_DIR = BASE_DIR / "training_data_words"
LINE_DIR = BASE_DIR / "training_data_lines" / "balanced_training"
OUTPUT_DIR = BASE_DIR / "combined_training"

# Settings
TARGET_HEIGHT = 64
WORD_SPACING = 30  # pixels between words
PADDING = 10  # pixels on left/right edges
MIN_WORDS_PER_LINE = 3
MAX_WORDS_PER_LINE = 5
NUM_SYNTHETIC_LINES = 30000  # How many synthetic lines to create from words
INCLUDE_REAL_LINES = False  # Set to False to only include synthetic lines
RANDOM_SEED = 42


def get_valid_pairs(folder, pattern="*.png"):
    """Get all image files that have corresponding .gt.txt with content."""
    pairs = []
    for png_path in sorted(folder.glob(pattern)):
        gt_path = png_path.with_suffix('.gt.txt')
        if gt_path.exists():
            try:
                text = gt_path.read_text(encoding='utf-8').strip()
                if text and len(text) > 0:
                    pairs.append((png_path, text))
            except:
                pass
    return pairs


def resize_to_height(img, target_height):
    """Resize image to target height, maintaining aspect ratio."""
    if img.height == target_height or img.height == 0:
        return img
    ratio = target_height / img.height
    new_width = max(1, int(img.width * ratio))
    return img.resize((new_width, target_height), Image.Resampling.LANCZOS)


def create_line_from_words(word_data_list, target_height, word_spacing, padding):
    """
    Concatenate multiple word images into a single line image.

    Args:
        word_data_list: List of (image_path, text) tuples
        target_height: Target height for the line
        word_spacing: Pixels between words
        padding: Pixels on left/right edges

    Returns:
        (PIL.Image, combined_text) or (None, None) if failed
    """
    # Load and resize all word images
    word_images = []
    texts = []

    for img_path, text in word_data_list:
        try:
            img = Image.open(img_path).convert('L')
            img = resize_to_height(img, target_height)
            if img.width > 0 and img.height > 0:
                word_images.append(img)
                texts.append(text)
        except Exception as e:
            continue

    if len(word_images) < 2:
        return None, None

    # Calculate total width
    total_width = padding * 2  # Left and right padding
    total_width += sum(img.width for img in word_images)
    total_width += word_spacing * (len(word_images) - 1)  # Spacing between words

    # Create the line image (white background)
    line_img = Image.new('L', (total_width, target_height), color=255)

    # Paste words from RIGHT to LEFT (RTL for Arabic/Persian)
    x_pos = total_width - padding
    for img in word_images:
        x_pos -= img.width
        line_img.paste(img, (x_pos, 0))
        x_pos -= word_spacing

    # Combine texts with spaces (RTL order)
    combined_text = ' '.join(texts)

    return line_img, combined_text


def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Creating Synthetic Lines from Word Images")
    print("=" * 60)

    # Check directories
    if not WORD_DIR.exists():
        print(f"ERROR: Word directory not found: {WORD_DIR}")
        sys.exit(1)

    # Load word data
    print(f"\nLoading word images from: {WORD_DIR}")
    word_pairs = get_valid_pairs(WORD_DIR)
    print(f"  Found {len(word_pairs)} valid word images")

    if len(word_pairs) < 100:
        print("ERROR: Not enough word images")
        sys.exit(1)

    # Load existing line data (only if including them)
    line_pairs = []
    if INCLUDE_REAL_LINES and LINE_DIR.exists():
        print(f"\nLoading line images from: {LINE_DIR}")
        line_pairs = get_valid_pairs(LINE_DIR)
        print(f"  Found {len(line_pairs)} valid line images")
    else:
        print(f"\nSkipping real lines (INCLUDE_REAL_LINES = False)")

    # Calculate how many synthetic lines to create
    num_synthetic = min(NUM_SYNTHETIC_LINES, len(word_pairs) // MIN_WORDS_PER_LINE)
    print(f"\nWill create {num_synthetic} synthetic lines from words")
    print(f"  Words per line: {MIN_WORDS_PER_LINE}-{MAX_WORDS_PER_LINE}")
    print(f"  Word spacing: {WORD_SPACING}px")

    total_images = len(line_pairs) + num_synthetic
    print(f"\nTotal combined dataset: {total_images} images")
    print(f"  - Real lines: {len(line_pairs)}")
    print(f"  - Synthetic lines: {num_synthetic}")

    # Check output directory
    if OUTPUT_DIR.exists():
        existing = len(list(OUTPUT_DIR.glob("*.png")))
        if existing > 0:
            print(f"\nOutput directory has {existing} images.")
            response = input("Delete and recreate? (y/n): ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return
            shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    idx = 0

    # Copy real line images first
    if line_pairs:
        print(f"\nCopying {len(line_pairs)} real line images...")
        for i, (png_path, text) in enumerate(line_pairs):
            dest_png = OUTPUT_DIR / f"line_{idx:06d}.png"
            dest_gt = OUTPUT_DIR / f"line_{idx:06d}.gt.txt"

            # Copy and standardize
            img = Image.open(png_path).convert('L')
            img = resize_to_height(img, TARGET_HEIGHT)
            img.save(dest_png)
            dest_gt.write_text(text, encoding='utf-8')

            idx += 1
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(line_pairs)}")

        print(f"  Done: {len(line_pairs)} real lines copied")

    # Create synthetic lines from words
    print(f"\nCreating {num_synthetic} synthetic lines...")

    # Shuffle word pairs for random sampling
    shuffled_words = word_pairs.copy()
    random.shuffle(shuffled_words)

    word_idx = 0
    created = 0
    failed = 0

    while created < num_synthetic and word_idx < len(shuffled_words) - MAX_WORDS_PER_LINE:
        # Pick random number of words for this line
        num_words = random.randint(MIN_WORDS_PER_LINE, MAX_WORDS_PER_LINE)

        # Get the next batch of words
        batch = shuffled_words[word_idx:word_idx + num_words]
        word_idx += num_words

        # Create synthetic line
        line_img, combined_text = create_line_from_words(
            batch, TARGET_HEIGHT, WORD_SPACING, PADDING
        )

        if line_img is not None and combined_text:
            dest_png = OUTPUT_DIR / f"line_{idx:06d}.png"
            dest_gt = OUTPUT_DIR / f"line_{idx:06d}.gt.txt"

            line_img.save(dest_png)
            dest_gt.write_text(combined_text, encoding='utf-8')

            idx += 1
            created += 1

            if created % 5000 == 0:
                print(f"  Created {created}/{num_synthetic} synthetic lines")
        else:
            failed += 1

        # Reset if we've used all words
        if word_idx >= len(shuffled_words) - MAX_WORDS_PER_LINE:
            random.shuffle(shuffled_words)
            word_idx = 0

    print(f"  Done: {created} synthetic lines created ({failed} failed)")

    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images: {idx}")
    print(f"  - Real lines: {len(line_pairs)}")
    print(f"  - Synthetic lines: {created}")
    print(f"\nTo train with this data:")
    print(f'  .\\train_combined.bat')


if __name__ == "__main__":
    main()
