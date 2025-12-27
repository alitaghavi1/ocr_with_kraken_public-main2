"""
Run Kraken line-level training with proper encoding for Windows.
"""

import os
import sys
import subprocess

# Set UTF-8 encoding to avoid Windows console issues
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'

# Training configuration
CONFIG = {
    'output': 'models/line_finetuned',
    'base_model': 'models/all_arabic_scripts.mlmodel',
    'training_data': 'training_data_lines/public_line_images/*.png',
    'batch_size': 8,
    'epochs': 50,
    'early_stopping': 10,
    'learning_rate': 0.0001,
    'device': 'cuda:0',
    'augment': False,  # Disabled due to dewarping issues
}

def main():
    print("=" * 60)
    print("Kraken Line-Level Fine-Tuning")
    print("=" * 60)
    print(f"Base model: {CONFIG['base_model']}")
    print(f"Training data: {CONFIG['training_data']}")
    print(f"Output: {CONFIG['output']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Device: {CONFIG['device']}")
    print("=" * 60)
    print()

    # Build command - use ketos.exe directly
    ketos_exe = os.path.join(os.path.dirname(sys.executable), 'ketos.exe')
    cmd = [
        ketos_exe,
        '-d', CONFIG['device'],
        'train',
        '-o', CONFIG['output'],
        '-f', 'path',
        '-B', str(CONFIG['batch_size']),
        '-N', str(CONFIG['epochs']),
        '--lag', str(CONFIG['early_stopping']),
        '-r', str(CONFIG['learning_rate']),
        '--schedule', 'reduceonplateau',
        '-i', CONFIG['base_model'],
        '--resize', 'union',
    ]

    if CONFIG['augment']:
        cmd.append('--augment')

    # Force binarization to handle different image modes
    cmd.append('--force-binarization')

    # Use legacy polygon extractor to avoid dewarping issues
    cmd.append('--legacy-polygons')

    cmd.append(CONFIG['training_data'])

    print("Running training...")
    print(f"Command: {' '.join(cmd[:10])}...")
    print()

    # Run with output to file to avoid encoding issues
    log_file = 'training_log.txt'

    with open(log_file, 'w', encoding='utf-8') as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'}
        )

        # Stream output to both console and file
        for line in process.stdout:
            # Try to print, ignore encoding errors
            try:
                print(line, end='', flush=True)
            except UnicodeEncodeError:
                print(line.encode('ascii', 'replace').decode(), end='', flush=True)
            log.write(line)
            log.flush()

        process.wait()

    print()
    print("=" * 60)
    if process.returncode == 0:
        print("Training completed successfully!")
        print(f"Model saved to: {CONFIG['output']}_best.mlmodel")
    else:
        print(f"Training failed with exit code: {process.returncode}")
        print(f"Check {log_file} for details")

    print(f"Full log saved to: {log_file}")
    return process.returncode


if __name__ == '__main__':
    sys.exit(main())
