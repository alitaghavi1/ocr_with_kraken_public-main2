"""
Prepare Kraken training data from labeled handwritten character/word dataset.

This script converts your labeled handwritten data to Kraken's training format.
It uses real handwritten images with their labels - much better than synthetic data.

Source: C:/AR/python_utils/training_output/prepared_data/
Output: Kraken-compatible training data (image + .gt.txt pairs)
"""

import os
import shutil
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = Path("C:/AR/python_utils/training_output/prepared_data")
OUTPUT_DIR = Path("C:/Ali/Handwritten_to_text/Source_Code/ocr_with_kraken_public-main/handwritten_training_data")

# Labels files
TRAIN_LABELS = SOURCE_DIR / "train_labels.txt"
VAL_LABELS = SOURCE_DIR / "val_labels.txt"

# Filter settings
MIN_LABEL_LENGTH = 1  # Minimum characters in label (1 = include single chars, 2 = words only)
MAX_LABEL_LENGTH = 50  # Maximum characters in label
MAX_SAMPLES = None  # Set to a number to limit samples (None = use all)

# Image settings
TARGET_HEIGHT = 48  # Resize images to this height (Kraken default)


def load_labels(labels_file, images_folder):
    """Load image paths and their labels"""
    data = []

    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                img_name, label = line.split('\t', 1)
                img_path = images_folder / img_name

                # Filter by label length
                if len(label) < MIN_LABEL_LENGTH or len(label) > MAX_LABEL_LENGTH:
                    continue

                if img_path.exists():
                    data.append((img_path, label))

    return data


def process_image(img_path, output_path):
    """Process and save image for Kraken training"""
    try:
        img = Image.open(img_path)

        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')

        # Resize to target height while maintaining aspect ratio
        if img.height != TARGET_HEIGHT:
            ratio = TARGET_HEIGHT / img.height
            new_width = int(img.width * ratio)
            img = img.resize((new_width, TARGET_HEIGHT), Image.LANCZOS)

        # Save
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False


def prepare_training_data():
    """Main function to prepare training data"""

    print("=" * 60)
    print("Preparing Kraken Training Data from Labeled Handwriting")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"\nLoading training labels from: {TRAIN_LABELS}")
    train_images_dir = SOURCE_DIR / "train" / "images"
    train_data = load_labels(TRAIN_LABELS, train_images_dir)
    print(f"  Found {len(train_data)} training samples")

    # Load validation data
    print(f"\nLoading validation labels from: {VAL_LABELS}")
    val_images_dir = SOURCE_DIR / "val" / "images"
    val_data = load_labels(VAL_LABELS, val_images_dir)
    print(f"  Found {len(val_data)} validation samples")

    # Combine all data
    all_data = train_data + val_data
    print(f"\nTotal samples: {len(all_data)}")

    # Limit samples if specified
    if MAX_SAMPLES and len(all_data) > MAX_SAMPLES:
        import random
        random.shuffle(all_data)
        all_data = all_data[:MAX_SAMPLES]
        print(f"Limited to {MAX_SAMPLES} samples")

    # Show label length distribution
    print("\nLabel length distribution:")
    length_counts = {}
    for _, label in all_data:
        l = len(label)
        length_counts[l] = length_counts.get(l, 0) + 1
    for length in sorted(length_counts.keys()):
        print(f"  {length} chars: {length_counts[length]} samples")

    # Process and save
    print(f"\nProcessing images...")
    print(f"Output directory: {OUTPUT_DIR}")

    success_count = 0
    for i, (img_path, label) in enumerate(all_data):
        # Output filenames
        img_filename = f"hw_{i:06d}.png"
        txt_filename = f"hw_{i:06d}.gt.txt"

        output_img_path = OUTPUT_DIR / img_filename
        output_txt_path = OUTPUT_DIR / txt_filename

        # Process image
        if process_image(img_path, output_img_path):
            # Write ground truth
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write(label)
            success_count += 1

        # Progress
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(all_data)} images...")

    print(f"\nDone! Successfully prepared {success_count} training samples")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Review a few samples to verify quality")
    print(f"  2. Run: .\\train_model.bat handwritten_training_data")


def show_samples(num_samples=5):
    """Show a few sample labels for verification"""
    print("\nSample labels from your data:")
    print("-" * 40)

    train_images_dir = SOURCE_DIR / "train" / "images"
    data = load_labels(TRAIN_LABELS, train_images_dir)

    import random
    samples = random.sample(data, min(num_samples, len(data)))

    for img_path, label in samples:
        print(f"  {img_path.name}: '{label}'")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        show_samples(10)
    else:
        prepare_training_data()
