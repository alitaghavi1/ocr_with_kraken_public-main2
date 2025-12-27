"""
Persian/Arabic OCR Script using Kraken
Usage: python run_ocr_batch.py <input_file>
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
TEMP_DIR = SCRIPT_DIR / "TEMP"
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.jp2')

# Model options
MODELS = {
    "fine-tuned": "models/fine_tuned_best.mlmodel",
    "old": "models/arabPers-WithDiffTypefaces.mlmodel",
}
DEFAULT_MODEL = "fine-tuned"

def setup_dirs():
    """Create necessary directories"""
    TEMP_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

def extract_images_from_pdf(pdf_path, output_folder):
    """Extract images from PDF using pdfimages (poppler)"""
    print(f"Extracting images from PDF: {pdf_path}")
    output_prefix = output_folder / "page"

    try:
        result = subprocess.run(
            ["pdfimages", "-png", str(pdf_path), str(output_prefix)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"pdfimages error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: pdfimages not found. Please install Poppler.")
        print("Download from: https://github.com/osudrl/poppler-windows/releases")
        return False

    return True

def run_ocr_on_image(img_path, output_path, model_str):
    """Run Kraken OCR on a single image"""
    print(f"Running OCR on: {img_path}")

    # Use relative paths from SCRIPT_DIR
    try:
        img_rel = img_path.relative_to(SCRIPT_DIR)
        out_rel = output_path.relative_to(SCRIPT_DIR)
    except ValueError:
        img_rel = img_path
        out_rel = output_path

    # Convert to forward slashes
    img_str = str(img_rel).replace('\\', '/')
    out_str = str(out_rel).replace('\\', '/')

    cmd = [
        "kraken",
        "-i", img_str, out_str,
        "binarize",
        "segment", "-d", "horizontal-rl", "-p", "20", "20",
        "ocr", "-m", model_str
    ]

    # Run with output redirected to avoid encoding issues
    log_file = SCRIPT_DIR / "ocr_log.txt"

    # Set environment to use UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    with open(log_file, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=log,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(SCRIPT_DIR),
            env=env
        )

    # Check if output file exists (OCR may succeed but crash on Unicode output)
    if output_path.exists():
        return True
    return result.returncode == 0

def process_file(input_path, model_choice):
    """Process a PDF or image file"""
    input_path = Path(input_path).resolve()
    model_str = MODELS.get(model_choice, MODELS[DEFAULT_MODEL])

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    basename = input_path.stem
    ext = input_path.suffix.lower()

    # Create temp folder for this file
    work_folder = TEMP_DIR / basename
    if work_folder.exists():
        shutil.rmtree(work_folder)
    work_folder.mkdir(parents=True)

    # Get list of images to process
    images = []

    if ext == '.pdf':
        # Extract images from PDF
        if not extract_images_from_pdf(input_path, work_folder):
            return False
        # Find extracted images
        images = sorted(work_folder.glob("page-*.png"))
        if not images:
            print("Error: No images extracted from PDF")
            return False
        print(f"Extracted {len(images)} page(s) from PDF")
    elif ext in IMG_EXTENSIONS:
        # Copy image to work folder
        dest = work_folder / f"page-000{ext}"
        shutil.copy(input_path, dest)
        images = [dest]
    else:
        print(f"Error: Unsupported file format: {ext}")
        print(f"Supported formats: PDF, {', '.join(IMG_EXTENSIONS)}")
        return False

    # Run OCR on each image
    all_text = []
    for i, img_path in enumerate(images):
        txt_path = img_path.with_suffix('.txt')

        if run_ocr_on_image(img_path, txt_path, model_str):
            if txt_path.exists():
                with open(txt_path, 'r', encoding='utf-8') as f:
                    page_text = f.read()
                all_text.append(page_text)
                if len(images) > 1:
                    all_text.append(f"\n\n--- Page {i+1} ---\n\n")
                print(f"  Page {i+1}/{len(images)} completed")
            else:
                print(f"  Warning: No output for page {i+1}")
        else:
            print(f"  Error processing page {i+1}")

    # Combine all pages and save to OUTPUT folder
    if all_text:
        final_output = OUTPUT_DIR / f"{basename}.txt"
        with open(final_output, 'w', encoding='utf-8') as f:
            f.write(''.join(all_text))
        print(f"\nOutput saved to: {final_output}")

        # Also display the text (with encoding protection)
        print("\n" + "="*50)
        print("OCR OUTPUT:")
        print("="*50)
        try:
            print(''.join(all_text))
        except UnicodeEncodeError:
            print("[Persian text saved to file - cannot display in console]")

        return True

    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_ocr_batch.py <input_file> [model]")
        print("Supported formats: PDF, JPG, JPEG, PNG, TIF, BMP")
        print(f"Model options: {', '.join(MODELS.keys())} (default: {DEFAULT_MODEL})")
        sys.exit(1)

    setup_dirs()

    input_file = sys.argv[1]
    model_choice = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print(f"Using model: {model_choice} ({MODELS.get(model_choice, MODELS[DEFAULT_MODEL])})")

    success = process_file(input_file, model_choice)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
