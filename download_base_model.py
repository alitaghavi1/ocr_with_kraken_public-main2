"""
Download Kraken Base Models for Fine-Tuning

This script downloads pre-trained Kraken models that you can fine-tune
with your handwritten data.

Usage:
    python download_base_model.py [model_name]
    python download_base_model.py --list   # Show available models
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"

# Popular base models for fine-tuning
RECOMMENDED_MODELS = {
    # Arabic/Persian models
    "arabic_best": "arabic_best.mlmodel",
    "arabPers": "arabPers-WithDiffTypefaces.mlmodel",

    # Generic/multilingual models
    #"en_best": "en_best.mlmodel",

    # Historical document models
    #"german_print": "german_print.mlmodel",
}


def list_available_models():
    """List all available models from Kraken model repository"""
    print("Fetching available models from Kraken repository...")
    print("=" * 60)

    # Use kraken to list models
    cmd = [sys.executable, "-m", "kraken", "list"]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error listing models:", result.stderr)
        print("\nTry running: kraken list")

    print("\n" + "=" * 60)
    print("Recommended models for handwritten Persian/Arabic:")
    print("  - arabic_best (general Arabic text)")
    print("  - Look for models containing 'persian', 'arabic', or 'handwritten'")
    print("\nTo download: python download_base_model.py <model_name>")


def download_model(model_name):
    """Download a model from Kraken repository"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Destination: {MODELS_DIR}")
    print("=" * 60)

    # Use kraken get to download
    cmd = [
        sys.executable, "-m", "kraken",
        "get", model_name
    ]

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    result = subprocess.run(cmd, env=env)

    if result.returncode == 0:
        print("\nModel downloaded successfully!")
        print(f"\nTo use for fine-tuning, update train.py:")
        print(f"  BASE_MODEL = 'path/to/{model_name}'")
    else:
        print("\nDownload failed. Check the model name and try again.")
        print("Use: python download_base_model.py --list")

    return result.returncode


def get_kraken_model_dir():
    """Find where kraken stores downloaded models"""
    try:
        from kraken import repo
        return repo.get_model_path()
    except:
        # Default locations
        if sys.platform == 'win32':
            return Path.home() / '.kraken'
        else:
            return Path.home() / '.kraken'


def copy_model_to_local(model_name):
    """Copy a downloaded model to the local models directory"""
    import shutil

    kraken_dir = get_kraken_model_dir()
    source = kraken_dir / model_name
    dest = MODELS_DIR / model_name

    if source.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)
        print(f"Copied {model_name} to {dest}")
        return True
    else:
        print(f"Model not found at {source}")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nQuick start:")
        print("  1. List available models:  python download_base_model.py --list")
        print("  2. Download a model:       python download_base_model.py arabic_best")
        print("\nNote: For handwritten text, you may need to fine-tune from scratch")
        print("      or use a generic model as starting point.")
        sys.exit(0)

    arg = sys.argv[1]

    if arg in ('--list', '-l', 'list'):
        list_available_models()
    else:
        download_model(arg)


if __name__ == "__main__":
    main()
