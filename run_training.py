"""
Kraken OCR Training Runner with Logging

This script runs training and writes all output to training_log.txt
to avoid Windows console encoding issues.

Usage:
    .venv\Scripts\python.exe run_training.py [options]

Options:
    --from-scratch    Train a new model from scratch
    --continue        Continue from last checkpoint
    --finetune        Fine-tune from a base model (default)
"""

import os
import sys
import glob
import subprocess
from datetime import datetime
from pathlib import Path

# Set working directory
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# Configuration - match train.py settings
TRAINING_MODE = "finetune"  # "finetune", "scratch", or "continue"
BASE_MODEL = None  # Set to path of base model for fine-tuning
CHECKPOINT_MODEL = "models/fine_tuned_best.mlmodel"
OUTPUT_PATH = "models/fine_tuned"
TRAINING_DATA_DIR = "handwritten_training_data"
LOG_FILE = "training_log.txt"

# Hyperparameters
BATCH_SIZE = 48
EPOCHS = 50
LEARNING_RATE = 0.0002
EARLY_STOPPING = 10
DEVICE = "cuda:0"


def main():
    # Parse mode from command line
    mode = TRAINING_MODE
    if '--from-scratch' in sys.argv or '--scratch' in sys.argv:
        mode = "scratch"
    elif '--continue' in sys.argv:
        mode = "continue"
    elif '--finetune' in sys.argv:
        mode = "finetune"

    # Find training files
    training_files = glob.glob(f'{TRAINING_DATA_DIR}/*.png')

    # Find ketos executable
    ketos_path = SCRIPT_DIR / '.venv' / 'Scripts' / 'ketos.exe'
    if not ketos_path.exists():
        ketos_path = 'ketos'  # Try system PATH

    # Build command
    cmd = [
        str(ketos_path),
        '-d', DEVICE,
        'train',
        '-o', OUTPUT_PATH,
        '-f', 'path',
        '-B', str(BATCH_SIZE),
        '-N', str(EPOCHS),
        '--lag', str(EARLY_STOPPING),
        '-r', str(LEARNING_RATE),
        '--augment',
        '--schedule', 'reduceonplateau',
        '-F', '1',
    ]

    # Add model based on mode
    if mode == "continue" and CHECKPOINT_MODEL and Path(CHECKPOINT_MODEL).exists():
        cmd.extend(['-i', CHECKPOINT_MODEL, '--resize', 'union'])
        model_info = f"Continuing from: {CHECKPOINT_MODEL}"
    elif mode == "finetune" and BASE_MODEL and Path(BASE_MODEL).exists():
        cmd.extend(['-i', BASE_MODEL, '--resize', 'union'])
        model_info = f"Fine-tuning from: {BASE_MODEL}"
    else:
        # Train from scratch - no resize option
        model_spec = "[1,120,0,1 Cr3,13,32 Do0.1,2 Mp2,2 Cr3,13,64 Do0.1,2 Mp2,2 S1(1x0)1,3 Lbx200 Do0.1,2 Lbx200 Do0.1,2 O2s1c{chars}]"
        cmd.extend(['--spec', model_spec])
        model_info = "Training from scratch"

    # Add training data
    cmd.append(f'{TRAINING_DATA_DIR}/*.png')

    # Set environment for UTF-8
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['TERM'] = 'dumb'

    # Open log file - all output goes here
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write("=" * 60 + "\n")
        log.write("Kraken OCR Fine-Tuning for Handwritten Data\n")
        log.write("=" * 60 + "\n")
        log.write(f"Training started: {datetime.now()}\n")
        log.write(f"Mode: {mode}\n")
        log.write(f"{model_info}\n")
        log.write(f"Training samples: {len(training_files)}\n")
        log.write(f"Device: {DEVICE}\n")
        log.write(f"Batch size: {BATCH_SIZE}\n")
        log.write(f"Learning rate: {LEARNING_RATE}\n")
        log.write(f"Max epochs: {EPOCHS}\n")
        log.write("=" * 60 + "\n\n")
        log.write(f"Command: {' '.join(cmd)}\n\n")
        log.flush()

        # Run training with output to log file
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Wait for process to complete
        process.wait()

        log.write(f"\n\nTraining finished: {datetime.now()}\n")
        log.write(f"Exit code: {process.returncode}\n")

    return process.returncode


if __name__ == '__main__':
    print("=" * 50)
    print("Kraken OCR Training")
    print("=" * 50)
    print("Training started. Check training_log.txt for progress.")
    print("Use 'tail -f training_log.txt' or open in editor to monitor.")
    print()

    exit_code = main()

    print()
    print("=" * 50)
    print(f"Training finished with exit code: {exit_code}")
    if exit_code == 0:
        print("Best model saved to: models/fine_tuned_best.mlmodel")
    print("See training_log.txt for details.")

    sys.exit(exit_code)
