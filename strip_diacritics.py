"""
Strip Arabic/Persian diacritics from ground truth files.

This creates a new training dataset without tashkeel (diacritical marks),
which makes OCR significantly easier. The consonantal skeleton is preserved.

Diacritics removed:
- Fatha, Damma, Kasra (short vowels)
- Fathatan, Dammatan, Kasratan (tanwin)
- Shadda (gemination)
- Sukun (no vowel)
- Superscript Alef, Maddah, Hamza marks, etc.

Usage:
    python strip_diacritics.py
"""

import os
import re
import shutil
from pathlib import Path

# Source and destination directories
SOURCE_DIR = Path("training_data_lines/balanced_training")
DEST_DIR = Path("training_data_lines/balanced_training_no_diacritics")

# Arabic diacritics (tashkeel) to remove
ARABIC_DIACRITICS = (
    '\u064B'  # Fathatan
    '\u064C'  # Dammatan
    '\u064D'  # Kasratan
    '\u064E'  # Fatha
    '\u064F'  # Damma
    '\u0650'  # Kasra
    '\u0651'  # Shadda
    '\u0652'  # Sukun
    '\u0653'  # Maddah above
    '\u0654'  # Hamza above
    '\u0655'  # Hamza below
    '\u0656'  # Subscript alef
    '\u0657'  # Inverted damma
    '\u0658'  # Mark noon ghunna
    '\u0659'  # Zwarakay
    '\u065A'  # Vowel sign small v above
    '\u065B'  # Vowel sign inverted small v above
    '\u065C'  # Vowel sign dot below
    '\u065D'  # Reversed damma
    '\u065E'  # Fatha with two dots
    '\u065F'  # Wavy hamza below
    '\u0670'  # Superscript alef
    '\u06D6'  # Small high ligature sad with lam with alef maksura
    '\u06D7'  # Small high ligature qaf with lam with alef maksura
    '\u06D8'  # Small high meem initial form
    '\u06D9'  # Small high lam alef
    '\u06DA'  # Small high jeem
    '\u06DB'  # Small high three dots
    '\u06DC'  # Small high seen
    '\u06DF'  # Small high rounded zero
    '\u06E0'  # Small high upright rectangular zero
    '\u06E1'  # Small high dotless head of khah
    '\u06E2'  # Small high meem isolated form
    '\u06E3'  # Small low seen
    '\u06E4'  # Small high madda
    '\u06E7'  # Small high yeh
    '\u06E8'  # Small high noon
    '\u06EA'  # Empty centre low stop
    '\u06EB'  # Empty centre high stop
    '\u06EC'  # Rounded high stop with filled centre
    '\u06ED'  # Small low meem
)

# Compile regex pattern for efficiency
DIACRITICS_PATTERN = re.compile('[' + ARABIC_DIACRITICS + ']')


def strip_diacritics(text: str) -> str:
    """Remove all Arabic diacritical marks from text."""
    return DIACRITICS_PATTERN.sub('', text)


def main():
    print("=" * 60)
    print("Strip Diacritics from Ground Truth Files")
    print("=" * 60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print()

    # Check source exists
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return 1

    # Create destination directory
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Find all ground truth files
    gt_files = list(SOURCE_DIR.glob("*.gt.txt"))
    print(f"Found {len(gt_files)} ground truth files")
    print()

    # Process files
    processed = 0
    errors = 0
    chars_removed = 0

    for gt_file in gt_files:
        try:
            # Read original ground truth
            original_text = gt_file.read_text(encoding='utf-8').strip()

            # Strip diacritics
            stripped_text = strip_diacritics(original_text)
            chars_removed += len(original_text) - len(stripped_text)

            # Write new ground truth
            new_gt_file = DEST_DIR / gt_file.name
            new_gt_file.write_text(stripped_text, encoding='utf-8')

            # Copy corresponding image (create symlink to save space, or copy)
            img_name = gt_file.stem.replace('.gt', '') + '.png'
            src_img = SOURCE_DIR / img_name
            dst_img = DEST_DIR / img_name

            if src_img.exists() and not dst_img.exists():
                # Use symlink on Windows (requires admin) or copy
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    # Symlink failed, copy instead
                    shutil.copy2(src_img, dst_img)

            processed += 1

            # Progress update
            if processed % 5000 == 0:
                print(f"Processed {processed}/{len(gt_files)} files...")

        except Exception as e:
            print(f"Error processing {gt_file}: {e}")
            errors += 1

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Files processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Diacritic characters removed: {chars_removed:,}")
    print(f"Output directory: {DEST_DIR}")
    print()
    print("Example transformation:")

    # Show example
    if gt_files:
        sample = gt_files[0]
        original = sample.read_text(encoding='utf-8').strip()
        stripped = strip_diacritics(original)
        print(f"  Original: {original}")
        print(f"  Stripped: {stripped}")

    print()
    print("Next steps:")
    print("  1. Create a new training batch file pointing to the new directory")
    print("  2. Run: train_lines_no_diacritics.bat")

    return 0


if __name__ == '__main__':
    exit(main())
