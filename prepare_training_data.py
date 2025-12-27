"""
Prepare Training Data for Kraken Fine-Tuning

This script helps you prepare training data for fine-tuning Kraken models.

Methods:
1. From existing OCR output - correct the OCR and use as ground truth
2. From scratch - manually transcribe line images

Usage:
    python prepare_training_data.py <image_file_or_folder>
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "models" / "arabPers-WithDiffTypefaces.mlmodel"

def segment_image(image_path, output_dir):
    """Segment a page image into line images"""
    print(f"Segmenting {image_path}...")

    # Use kraken to segment
    cmd = [
        "kraken", "-i", str(image_path), str(output_dir / "lines.json"),
        "binarize", "segment", "-d", "horizontal-rl", "-bl"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0

def extract_lines(image_path, output_dir):
    """Extract individual lines from a page image using kraken"""
    from PIL import Image
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, get the segmentation
    json_path = output_dir / "segmentation.json"

    cmd = [
        "kraken", "-i", str(image_path), str(json_path),
        "binarize", "segment", "-d", "horizontal-rl", "-bl"
    ]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if not json_path.exists():
        print("Segmentation failed. Using manual line extraction...")
        return False

    return True

def create_ground_truth_template(image_folder, output_folder):
    """
    Create ground truth templates from existing OCR output.
    Run OCR first, then you can correct the .gt.txt files manually.
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    images = [f for f in image_folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Found {len(images)} images")

    for img_path in images:
        # Run OCR on the image
        txt_path = output_folder / f"{img_path.stem}.gt.txt"
        temp_txt = output_folder / f"{img_path.stem}_temp.txt"

        # Copy image to output folder
        import shutil
        dest_img = output_folder / img_path.name
        if not dest_img.exists():
            shutil.copy(img_path, dest_img)

        # Run OCR to get initial transcription
        cmd = [
            "kraken", "-i", str(dest_img), str(temp_txt),
            "binarize", "segment", "-d", "horizontal-rl",
            "ocr", "-m", str(MODEL_PATH)
        ]

        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'

        print(f"Processing {img_path.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Rename to .gt.txt
        if temp_txt.exists():
            temp_txt.rename(txt_path)
            print(f"  Created: {txt_path.name}")
        else:
            # Create empty file
            txt_path.write_text("", encoding='utf-8')
            print(f"  Created empty: {txt_path.name}")

    print(f"\nGround truth templates created in: {output_folder}")
    print("\nNext steps:")
    print("1. Open each .gt.txt file in a text editor (VS Code, Notepad++)")
    print("2. Correct any OCR errors")
    print("3. Run: train_model.bat", output_folder)

def print_usage():
    print("""
Kraken Training Data Preparation Tool
======================================

Usage:
    python prepare_training_data.py <command> <args>

Commands:
    prepare <image_folder> <output_folder>
        - Runs OCR on images and creates .gt.txt templates
        - You then manually correct the .gt.txt files
        - Example: python prepare_training_data.py prepare my_scans training_data

    info
        - Shows information about training data format

Example Workflow:
    1. Put your page/line images in a folder (e.g., 'my_scans')
    2. Run: python prepare_training_data.py prepare my_scans training_data
    3. Edit the .gt.txt files in 'training_data' to correct errors
    4. Run: train_model.bat training_data
""")

def print_format_info():
    print("""
Training Data Format for Kraken
================================

Option 1: Image + Ground Truth Text
------------------------------------
For each image, create a .gt.txt file with the same name:

    training_data/
        line001.png
        line001.gt.txt    <- Contains the correct transcription
        line002.png
        line002.gt.txt
        ...

The .gt.txt file should contain the exact text in the image.
For multi-line images, each line in the .gt.txt corresponds to a line in the image.

Option 2: ALTO XML Format
--------------------------
    training_data/
        page001.png
        page001.xml       <- ALTO XML with line coordinates and text
        ...

Option 3: PageXML Format
-------------------------
    training_data/
        page001.png
        page001.xml       <- PageXML with line coordinates and text
        ...

Tips:
- Use line-level images for best results (one line of text per image)
- Ensure images are high quality (300+ DPI)
- Use PNG or TIFF format (lossless)
- Include diverse samples (different words, characters, styles)
- Aim for 100-500 lines minimum for fine-tuning
""")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "info":
        print_format_info()
    elif command == "prepare":
        if len(sys.argv) < 4:
            print("Usage: python prepare_training_data.py prepare <image_folder> <output_folder>")
            sys.exit(1)
        create_ground_truth_template(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
