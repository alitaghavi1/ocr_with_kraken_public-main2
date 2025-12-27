"""
OCR Image Script - Extract text from images using trained Kraken model

Usage:
    python ocr_image.py <image_path> [output_path] [--model MODEL]

Examples:
    python ocr_image.py my_image.jpg
    python ocr_image.py my_image.jpg result.txt
    python ocr_image.py my_image.jpg --model models/fine_tuned_17.mlmodel
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path


# Default settings
DEFAULT_MODEL = "models/fine_tuned_best.mlmodel"
KRAKEN_PATH = r".venv\Scripts\kraken.exe"


def ocr_image(image_path, output_path=None, model_path=DEFAULT_MODEL, direction="horizontal-rl", padding=20):
    """
    Run OCR on an image file using a trained Kraken model.

    Args:
        image_path: Path to the input image (jpg, png, tif, etc.)
        output_path: Path where the text output will be saved (optional)
        model_path: Path to the trained model
        direction: Text direction ('horizontal-rl' for right-to-left, 'horizontal-lr' for left-to-right)
        padding: Padding around detected lines

    Returns:
        str: The extracted text, or None if OCR failed
    """
    image_path = str(image_path)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Available models in 'models/' folder:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".mlmodel"):
                    print(f"  - models/{f}")
        return None

    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_ocr.txt"

    output_path = str(output_path)

    # Build kraken command
    cmd = [
        KRAKEN_PATH,
        "-i", image_path, output_path,
        "binarize",
        "segment", "-d", direction, "-p", str(padding), str(padding),
        "ocr", "-m", model_path
    ]

    print(f"Processing: {image_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print("-" * 50)

    # Set environment for UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    # Run kraken
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        if result.returncode != 0:
            print(f"Kraken error (exit code {result.returncode}):")
            if result.stderr:
                print(result.stderr)
            return None

        # Read and return the output
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print("OCR completed successfully!")
            print("-" * 50)
            return text
        else:
            print("Error: Output file was not created")
            return None

    except FileNotFoundError:
        print(f"Error: Kraken executable not found at {KRAKEN_PATH}")
        print("Make sure you have activated the virtual environment.")
        return None
    except Exception as e:
        print(f"Error running OCR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from images using trained Kraken OCR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_image.py document.jpg
  python ocr_image.py document.jpg output.txt
  python ocr_image.py document.jpg --model models/fine_tuned_17.mlmodel
  python ocr_image.py document.jpg -d horizontal-lr  (for left-to-right text)
        """
    )

    parser.add_argument("image", help="Path to the input image file (jpg, png, tif, etc.)")
    parser.add_argument("output", nargs="?", default=None, help="Path for output text file (optional)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Path to OCR model (default: {DEFAULT_MODEL})")
    parser.add_argument("-d", "--direction", default="horizontal-rl",
                        choices=["horizontal-rl", "horizontal-lr", "vertical-lr", "vertical-rl"],
                        help="Text direction (default: horizontal-rl for right-to-left)")
    parser.add_argument("-p", "--padding", type=int, default=20, help="Line padding (default: 20)")
    parser.add_argument("--print", dest="print_output", action="store_true",
                        help="Print the OCR result to console")

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run OCR
    text = ocr_image(
        args.image,
        args.output,
        args.model,
        args.direction,
        args.padding
    )

    if text is not None:
        if args.print_output:
            print("\n=== OCR Result ===\n")
            try:
                print(text)
            except UnicodeEncodeError:
                print("[Text contains characters that cannot be displayed in console]")
                print("Check the output file for full results.")

        # Show output file location
        output_path = args.output or f"{os.path.splitext(args.image)[0]}_ocr.txt"
        print(f"\nText saved to: {output_path}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
