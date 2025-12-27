"""
Simple OCR script that runs Kraken and saves output to file
Bypasses Windows console encoding issues
"""
import subprocess
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

img_path = "TEMP/yamini/yamini_000.png"
output_path = "TEMP/yamini/yamini_000.txt"
model_path = "models/arabPers-WithDiffTypefaces.mlmodel"

# Run kraken command and capture output
cmd = [
    "kraken",
    "-i", img_path, output_path,
    "binarize",
    "segment", "-d", "horizontal-rl", "-p", "20", "20",
    "ocr", "-m", model_path
]

print(f"Running: {' '.join(cmd)}")

# Run with output redirected to file to avoid encoding issues
with open("ocr_log.txt", "w", encoding="utf-8") as log_file:
    result = subprocess.run(
        cmd,
        stdout=log_file,
        stderr=log_file,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

print("Return code:", result.returncode)

# Check if output file was created
if os.path.exists(output_path):
    print(f"\nOutput file created: {output_path}")
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Write to a separate file to preserve Persian characters
    with open("ocr_result.txt", 'w', encoding='utf-8') as f:
        f.write(content)
    print("\n=== OCR OUTPUT (saved to ocr_result.txt) ===")
    # Try to print, but handle encoding errors
    try:
        print(content)
    except UnicodeEncodeError:
        print("[Content contains characters that cannot be displayed in console]")
        print("Please check ocr_result.txt for the full output")
else:
    print("Output file was NOT created")
    print("Check ocr_log.txt for error details")
