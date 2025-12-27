"""
Convert character-level dataset to line-level data for Kraken training.

This script creates synthetic line images by combining individual character images,
which can then be used to fine-tune Kraken OCR models.
"""

import os
import random
from pathlib import Path
from PIL import Image
import json

# Configuration
CHAR_DATASET_DIR = Path("C:/AR/python_utils/training_output/prepared_data")
OUTPUT_DIR = Path("C:/Ali/Handwritten_to_text/Source_Code/ocr_with_kraken_public-main/synthetic_training_data")
LABELS_FILE = CHAR_DATASET_DIR / "train_labels.txt"

# Line generation settings
MIN_CHARS_PER_LINE = 5
MAX_CHARS_PER_LINE = 15
NUM_LINES_TO_GENERATE = 5000
LINE_HEIGHT = 48  # pixels
CHAR_SPACING = 0   # pixels between characters within a word (touching)
WORD_SPACING = 60  # pixels between words (clear visible gap)
MIN_CHARS_PER_WORD = 1
MAX_CHARS_PER_WORD = 5
PADDING = 10  # pixels on each side

def autocrop_image(img):
    """Remove whitespace padding around the character"""
    from PIL import ImageOps

    # Invert (so character is white, background is black)
    inverted = ImageOps.invert(img.convert('L'))
    # Get bounding box of non-zero pixels
    bbox = inverted.getbbox()
    if bbox:
        # Add small padding (2px) so chars don't touch edges
        left, top, right, bottom = bbox
        left = max(0, left - 2)
        top = max(0, top - 2)
        right = min(img.width, right + 2)
        bottom = min(img.height, bottom + 2)
        return img.crop((left, top, right, bottom))
    return img

def load_character_data():
    """Load character images and their labels"""
    print("Loading character data...")

    char_data = []  # List of (image_path, label)

    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                img_name, label = line.split('\t', 1)
                img_path = CHAR_DATASET_DIR / "train" / "images" / img_name
                if img_path.exists():
                    char_data.append((img_path, label))

    print(f"Loaded {len(char_data)} characters")
    return char_data

def create_line_image(words):
    """
    Combine character images into a single line image with words separated by spaces.

    Args:
        words: List of words, where each word is a list of (image, label) tuples

    Returns:
        (line_image, ground_truth_text)
    """

    # Calculate total width
    total_width = PADDING * 2
    for word in words:
        for img, label in word:
            total_width += img.width + CHAR_SPACING
        total_width -= CHAR_SPACING  # Remove last char spacing in word
        total_width += WORD_SPACING  # Add word spacing
    total_width -= WORD_SPACING  # Remove last word spacing

    # Create new image (white background)
    line_img = Image.new('L', (total_width, LINE_HEIGHT), color=255)

    # Paste characters from RIGHT to LEFT (for RTL text)
    x_pos = total_width - PADDING

    word_labels = []
    for word in words:
        word_text = ''
        for img, label in word:
            # Center vertically
            y_pos = (LINE_HEIGHT - img.height) // 2
            x_pos -= img.width
            line_img.paste(img, (x_pos, y_pos))
            x_pos -= CHAR_SPACING
            word_text += label
        x_pos += CHAR_SPACING  # Undo last char spacing
        x_pos -= WORD_SPACING  # Add word gap
        word_labels.append(word_text)

    # Combine word labels with spaces
    text = ' '.join(word_labels)

    return line_img, text

def generate_synthetic_lines(char_data, num_lines):
    """Generate synthetic line images with words separated by spaces"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_lines} synthetic lines...")

    generated = []

    for i in range(num_lines):
        # Random number of total characters
        num_chars = random.randint(MIN_CHARS_PER_LINE, MAX_CHARS_PER_LINE)

        # Select random characters
        selected = random.sample(char_data, min(num_chars, len(char_data)))

        # Load images and group into words
        all_chars = []
        for img_path, label in selected:
            try:
                img = Image.open(img_path)
                # Auto-crop whitespace
                img = autocrop_image(img)
                # Resize if needed to fit line height
                if img.height > LINE_HEIGHT - 10:
                    ratio = (LINE_HEIGHT - 10) / img.height
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                all_chars.append((img, label))
            except Exception as e:
                continue

        if len(all_chars) < 3:
            continue

        # Group characters into words
        words = []
        idx = 0
        while idx < len(all_chars):
            word_len = random.randint(MIN_CHARS_PER_WORD, MAX_CHARS_PER_WORD)
            word = all_chars[idx:idx + word_len]
            if word:
                words.append(word)
            idx += word_len

        if len(words) < 1:
            continue

        # Create line image
        line_img, text = create_line_image(words)

        # Save
        img_filename = f"line_{i:06d}.png"
        txt_filename = f"line_{i:06d}.gt.txt"

        line_img.save(OUTPUT_DIR / img_filename)
        with open(OUTPUT_DIR / txt_filename, 'w', encoding='utf-8') as f:
            f.write(text)

        generated.append((img_filename, text))

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_lines} lines")

    print(f"\nGenerated {len(generated)} synthetic lines")
    print(f"Output directory: {OUTPUT_DIR}")
    return generated

def verify_samples(char_data, num_samples=5):
    """Generate a few samples for visual verification before full generation"""
    verify_dir = OUTPUT_DIR / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {num_samples} verification samples...")
    print("Please check these manually before full generation!\n")

    for i in range(num_samples):
        num_chars = random.randint(MIN_CHARS_PER_LINE, MAX_CHARS_PER_LINE)
        selected = random.sample(char_data, min(num_chars, len(char_data)))

        # Load images
        all_chars = []
        for img_path, label in selected:
            try:
                img = Image.open(img_path)
                # Auto-crop whitespace
                img = autocrop_image(img)
                if img.height > LINE_HEIGHT - 10:
                    ratio = (LINE_HEIGHT - 10) / img.height
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                all_chars.append((img, label))
            except:
                continue

        if len(all_chars) < 3:
            continue

        # Group into words
        words = []
        idx = 0
        while idx < len(all_chars):
            word_len = random.randint(MIN_CHARS_PER_WORD, MAX_CHARS_PER_WORD)
            word = all_chars[idx:idx + word_len]
            if word:
                words.append(word)
            idx += word_len

        if len(words) < 1:
            continue

        line_img, text = create_line_image(words)

        img_filename = f"verify_{i:02d}.png"
        txt_filename = f"verify_{i:02d}.gt.txt"

        line_img.save(verify_dir / img_filename)
        with open(verify_dir / txt_filename, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"  Sample {i+1}: {img_filename}")
        print(f"    Ground truth: {text}")
        print()

    print(f"\nVerification samples saved to: {verify_dir}")
    print("Please open the images and check if the text matches!")
    print("\nPress Enter to continue with full generation, or Ctrl+C to abort...")
    input()

def main():
    print("=" * 50)
    print("Character to Line Converter for Kraken Training")
    print("=" * 50)

    # Load character data
    char_data = load_character_data()

    if not char_data:
        print("No character data found!")
        return

    # First verify a few samples
    verify_samples(char_data, num_samples=5)

    # Generate synthetic lines
    generated = generate_synthetic_lines(char_data, NUM_LINES_TO_GENERATE)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"1. Review the generated lines in: {OUTPUT_DIR}")
    print(f"2. Correct any errors in the .gt.txt files")
    print(f"3. Run: train_model.bat {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
