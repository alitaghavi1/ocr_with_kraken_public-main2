"""
Download Arabic OCR datasets from Hugging Face

Multiple datasets available:
1. mssqpi/Arabic-OCR-Dataset - General Arabic OCR
2. arabic-img2md - Arabic document images (for Nougat model)

Uses huggingface_hub for downloading.
"""

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from PIL import Image

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "hf_arabic_lines"


def download_mssqpi_dataset():
    """Download mssqpi/Arabic-OCR-Dataset from Hugging Face."""
    print("\n" + "=" * 50)
    print("Downloading mssqpi/Arabic-OCR-Dataset")
    print("=" * 50)

    repo_id = "mssqpi/Arabic-OCR-Dataset"

    try:
        # Load dataset
        print("Loading dataset (this may take a while)...")
        dataset = load_dataset(repo_id, split="train")

        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Features: {dataset.features}")

        # Check structure
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")

        # Process and save
        total_saved = 0
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(dataset):
            try:
                # Try different possible column names
                img = None
                text = None

                # Image columns
                for col in ['image', 'img', 'pixel_values', 'input_image']:
                    if col in sample and sample[col] is not None:
                        img = sample[col]
                        break

                # Text columns
                for col in ['text', 'label', 'transcription', 'ground_truth', 'target']:
                    if col in sample and sample[col] is not None:
                        text = sample[col]
                        break

                if img is None or text is None:
                    continue

                # Convert image if needed
                if not isinstance(img, Image.Image):
                    continue

                # Convert to grayscale
                if img.mode != 'L':
                    img = img.convert('L')

                # Save
                out_png = OUTPUT_DIR / f"hf_arabic_{total_saved:05d}.png"
                out_gt = OUTPUT_DIR / f"hf_arabic_{total_saved:05d}.gt.txt"

                img.save(out_png)
                out_gt.write_text(str(text).strip(), encoding='utf-8')
                total_saved += 1

                if total_saved % 500 == 0:
                    print(f"  Saved {total_saved} samples...")

            except Exception as e:
                if i < 5:
                    print(f"  Error on sample {i}: {e}")

        print(f"Saved {total_saved} samples to {OUTPUT_DIR}")
        return total_saved

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0


def list_arabic_datasets():
    """List known Arabic OCR datasets on Hugging Face."""
    datasets = [
        {
            "name": "mssqpi/Arabic-OCR-Dataset",
            "description": "General Arabic OCR dataset",
            "type": "dataset"
        },
        {
            "name": "MohamedRashad/arabic-img2md",
            "description": "Arabic document images for OCR",
            "type": "dataset"
        },
        {
            "name": "johnlockejrr/KHATT_v1.0_dataset",
            "description": "KHATT Arabic handwriting dataset",
            "type": "dataset"
        },
        {
            "name": "Muharaf/Muharaf",
            "description": "Historical handwritten Arabic manuscripts",
            "type": "dataset"
        },
    ]

    print("\nKnown Arabic OCR Datasets on Hugging Face:")
    print("-" * 60)
    for ds in datasets:
        print(f"  {ds['name']}")
        print(f"    {ds['description']}")
    print("-" * 60)

    return datasets


def main():
    print("=" * 60)
    print("Hugging Face Arabic OCR Dataset Downloader")
    print("=" * 60)

    if not HF_AVAILABLE:
        print("\nError: huggingface_hub and datasets packages required.")
        print("Install with: pip install huggingface_hub datasets")
        return

    # List available datasets
    datasets = list_arabic_datasets()

    # Try to download the main dataset
    print("\nAttempting to download Arabic OCR datasets...")

    total = download_mssqpi_dataset()

    print()
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"  Total samples downloaded: {total}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if total == 0:
        print("\nNote: Dataset structure may have changed.")
        print("Try downloading manually from Hugging Face.")


if __name__ == "__main__":
    main()
