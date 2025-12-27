"""
Kraken OCR Fine-Tuning Script

This script fine-tunes a Kraken OCR model with your handwritten data.

Training Modes:
1. Fine-tune from base model: Use a pre-trained Kraken model as starting point
2. Train from scratch: Build a new model for your specific handwriting
3. Continue training: Resume from a previous checkpoint

Usage:
    .venv\Scripts\python.exe train.py                    # Use config below
    .venv\Scripts\python.exe train.py --from-scratch     # Train new model
    .venv\Scripts\python.exe train.py --continue         # Continue from checkpoint
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['TERM'] = 'dumb'

# Change to script directory
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# =============================================================================
# TRAINING CONFIGURATION - Edit these settings
# =============================================================================

# Training mode: "finetune", "scratch", or "continue"
TRAINING_MODE = "finetune"

# Base model for fine-tuning (downloaded Kraken model or your own)
# Set to None to train from scratch
BASE_MODEL = 'models/all_arabic_scripts.mlmodel'  # e.g., "models/arabic_best.mlmodel"
# BASE_MODEL = 'models/line_finetuned_5.mlmodel.mlmodel'  # e.g., "models/arabic_best.mlmodel"


# Checkpoint to continue from (only used if TRAINING_MODE = "continue")
CHECKPOINT_MODEL = "models/fine_tuned_best.mlmodel"

# Output model path (will add version numbers: fine_tuned_0.mlmodel, etc.)
OUTPUT_PATH = "models/fine_tuned"

# Training data directory (should contain .png + .gt.txt pairs)
TRAINING_DATA_DIR = "training_data_lines/balanced_training"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 16          # Reduce if GPU memory issues (try 8 or 4)
EPOCHS = 50              # Maximum training epochs
LEARNING_RATE = 0.0001   # Learning rate (lower for fine-tuning: 0.0001)
EARLY_STOPPING = 10      # Stop if no improvement for N epochs
VALIDATION_FREQ = 1      # Validate every N epochs
DEVICE = "cuda:0"        # Use "cpu" if no GPU available

# Data augmentation (recommended for handwritten data)
USE_AUGMENTATION = True

# Learning rate schedule
LR_SCHEDULE = "reduceonplateau"  # or "cosine", "exponential", "1cycle"

# Image resize mode: "union", "new", "fail", "both"
RESIZE_MODE = "new"

# Model architecture for training from scratch (Kraken spec format)
# This is used when BASE_MODEL is None
MODEL_SPEC = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,64 Do0.1,2 Mp2,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 O2s1c{chars}]"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def find_training_files():
    """Find all training image files"""
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(TRAINING_DATA_DIR, pattern)))
    return files


def validate_training_data(files):
    """Check that ground truth files exist for training images"""
    valid = 0
    missing = []

    for f in files:
        gt_file = Path(f).with_suffix('.gt.txt')
        if gt_file.exists():
            valid += 1
        else:
            missing.append(f)

    return valid, missing


def build_training_command(mode, training_files):
    """Build the ketos training command"""
    ketos = str(SCRIPT_DIR / ".venv" / "Scripts" / "ketos.exe")
    if not Path(ketos).exists():
        ketos = sys.executable + " -m kraken.ketos"

    cmd = [
        ketos,
        '-d', DEVICE,
        'train',
        '-o', OUTPUT_PATH,
        '-f', 'path',
        '-B', str(BATCH_SIZE),
        '-N', str(EPOCHS),
        '--lag', str(EARLY_STOPPING),
        '-r', str(LEARNING_RATE),
        '--schedule', LR_SCHEDULE,
        '-F', str(VALIDATION_FREQ),
    ]

    # Add augmentation if enabled
    if USE_AUGMENTATION:
        cmd.append('--augment')

    # Add model based on mode
    if mode == "continue" and CHECKPOINT_MODEL:
        if not Path(CHECKPOINT_MODEL).exists():
            print(f"ERROR: Checkpoint not found: {CHECKPOINT_MODEL}")
            sys.exit(1)
        cmd.extend(['-i', CHECKPOINT_MODEL, '--resize', RESIZE_MODE])
        print(f"Continuing from: {CHECKPOINT_MODEL}")

    elif mode == "finetune" and BASE_MODEL:
        if not Path(BASE_MODEL).exists():
            print(f"ERROR: Base model not found: {BASE_MODEL}")
            print("Download a base model first:")
            print("  python download_base_model.py --list")
            print("  python download_base_model.py <model_name>")
            sys.exit(1)
        cmd.extend(['-i', BASE_MODEL, '--resize', RESIZE_MODE])
        print(f"Fine-tuning from: {BASE_MODEL}")

    else:
        # Training from scratch - no resize option needed
        print("Training from scratch with custom architecture")
        cmd.extend(['--spec', MODEL_SPEC])

    # Add training files
    cmd.append(f'{TRAINING_DATA_DIR}/*.png')

    return cmd


def main():
    print("=" * 60)
    print("Kraken OCR Fine-Tuning for Handwritten Data")
    print("=" * 60)

    # Parse command line arguments
    mode = TRAINING_MODE
    if '--from-scratch' in sys.argv or '--scratch' in sys.argv:
        mode = "scratch"
    elif '--continue' in sys.argv:
        mode = "continue"
    elif '--finetune' in sys.argv:
        mode = "finetune"

    print(f"Training mode: {mode}")

    # Find and validate training data
    training_files = find_training_files()
    print(f"Found {len(training_files)} training images")

    if len(training_files) == 0:
        print(f"\nERROR: No training images found in {TRAINING_DATA_DIR}/")
        print("\nTo prepare training data:")
        print("1. Create image + text pairs:")
        print("   - image001.png + image001.gt.txt")
        print("   - image002.png + image002.gt.txt")
        print("2. Each .gt.txt contains the exact text in the image")
        print("\nOr use: python prepare_training_data.py prepare <folder> <output>")
        sys.exit(1)

    valid, missing = validate_training_data(training_files)
    print(f"Valid pairs (image + ground truth): {valid}")

    if missing:
        print(f"WARNING: {len(missing)} images missing ground truth files")
        if len(missing) <= 5:
            for m in missing:
                print(f"  - {m}")

    if valid == 0:
        print("\nERROR: No valid training pairs found!")
        print("Each image needs a .gt.txt file with the same name")
        sys.exit(1)

    print(f"Output: {OUTPUT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {EPOCHS}")
    print(f"Early stopping: {EARLY_STOPPING} epochs")
    print("=" * 60)

    # Build command
    cmd = build_training_command(mode, training_files)
    print(f"\nCommand: {' '.join(cmd[:10])}...")
    print("\nStarting training...\n")

    # Run training
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))

    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("Training completed successfully!")
        print(f"Best model saved to: {OUTPUT_PATH}_best.mlmodel")
    else:
        print(f"Training failed with exit code: {result.returncode}")

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
