"""
Run Kraken training with balanced dataset from manifest file.
"""

import subprocess
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MANIFEST_FILE = BASE_DIR / "training_manifest.txt"
OUTPUT_MODEL = BASE_DIR / "models" / "line_finetuned"
BASE_MODEL = BASE_DIR / "models" / "all_arabic_scripts.mlmodel"


def main():
    # Read manifest file
    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file not found: {MANIFEST_FILE}")
        print("Run create_balanced_manifest.py first")
        sys.exit(1)

    with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"Training with {len(files)} files from manifest")

    # Build ketos command
    ketos_path = BASE_DIR / ".venv" / "Scripts" / "ketos.exe"

    cmd = [
        str(ketos_path),
        "-d", "cuda:0",
        "train",
        "-o", str(OUTPUT_MODEL),
        "-f", "path",
        "-B", "8",
        "-N", "50",
        "--lag", "10",
        "-r", "0.0001",
        "--schedule", "reduceonplateau",
        "--augment",
        "-i", str(BASE_MODEL),
        "--resize", "union",
    ] + files

    print(f"Running ketos with {len(files)} training files...")
    print()

    # Run training
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
