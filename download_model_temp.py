"""
Download standard Kraken Arabic/Persian OCR models from Zenodo

Available models:
  - Arabic-Script Base Model (Arabic, Persian, Urdu, Ottoman Turkish): 10.5281/zenodo.7050270
  - Persian-specific fine-tuned model: 10.5281/zenodo.10788590

Usage:
  python download_model_temp.py              # Download Arabic-Persian base model
  python download_model_temp.py persian      # Download Persian-specific model
"""

import requests
import time
import os
import sys

# Zenodo record IDs
MODELS = {
    "arabic_persian": "7050270",  # Arabic-script base model (includes Persian)
    "persian": "10788590",        # Persian-specific fine-tuned model
}

def download_from_zenodo(record_id, output_dir="models"):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://zenodo.org/api/records/{record_id}"

    print(f"Fetching record info from Zenodo ({record_id})...")

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        print(f"Record: {data.get('metadata', {}).get('title', 'N/A')}")
        print("\nFiles available:")

        for f in data.get("files", []):
            key = f["key"]
            size_mb = f["size"]/1024/1024
            link = f["links"]["self"]
            print(f"  - {key}: {size_mb:.1f} MB")

            # Download the .mlmodel file
            if key.endswith(".mlmodel"):
                model_path = os.path.join(output_dir, key)
                if os.path.exists(model_path):
                    print(f"\n{model_path} already exists, skipping...")
                    return model_path

                print(f"\nDownloading {key}...")
                print(f"URL: {link}")

                resp = requests.get(link, headers=headers, stream=True, timeout=300)
                resp.raise_for_status()

                total = int(resp.headers.get('content-length', 0))
                downloaded = 0

                with open(model_path, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=8192):
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 / total
                            print(f"\rProgress: {pct:.1f}% ({downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB)", end="")

                print(f"\n\nSaved to: {model_path}")
                return model_path

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("\n*** RATE LIMITED ***")
            print("Zenodo is limiting requests. Please wait 1-2 minutes and try again.")
            print("Or download manually from:")
            print(f"  https://zenodo.org/records/{record_id}")
        else:
            print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

    return None


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # Choose model based on argument
    model_type = sys.argv[1] if len(sys.argv) > 1 else "arabic_persian"

    if model_type not in MODELS:
        print(f"Unknown model type: {model_type}")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    record_id = MODELS[model_type]
    model_path = download_from_zenodo(record_id)

    if model_path:
        print(f"\n\nTo use this model, run:")
        print(f'  python ocr_image.py examples\\yamini_full.jpg --model "{model_path}" --print')
