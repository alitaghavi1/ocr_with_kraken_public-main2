"""
Filter out images that cause dewarping errors in Kraken training.
"""

import os
import glob
import shutil
from PIL import Image
import numpy as np

# Try to import Kraken's lineest for testing
try:
    from kraken.lib.lineest import CenterNormalizer
    HAS_KRAKEN = True
except ImportError:
    HAS_KRAKEN = False
    print("Warning: kraken not available, using basic checks only")

SOURCE_DIR = "training_data_lines/public_line_images"
OUTPUT_DIR = "training_data_lines/public_line_images_filtered"
BAD_DIR = "training_data_lines/public_line_images_bad"


def test_image_dewarp(img_path):
    """Test if an image can be processed by Kraken's dewarping."""
    try:
        im = Image.open(img_path)

        # Convert to grayscale if needed
        if im.mode not in ['L', '1']:
            im = im.convert('L')

        arr = np.array(im)

        # Basic checks
        if arr.shape[0] < 10 or arr.shape[1] < 20:
            return False, "too_small"

        if HAS_KRAKEN:
            # Test with Kraken's normalizer
            normalizer = CenterNormalizer()
            try:
                # This is what fails in training
                cval = np.amax(arr)
                normalized = normalizer.normalize(arr, cval=cval)
                return True, "ok"
            except ValueError as e:
                return False, f"dewarp_error: {str(e)[:50]}"
            except Exception as e:
                return False, f"error: {str(e)[:50]}"
        else:
            # Basic numpy check for array consistency
            try:
                # Check if array is consistent
                _ = np.array(arr, dtype=np.float32)
                return True, "ok"
            except:
                return False, "array_error"

    except Exception as e:
        return False, f"load_error: {str(e)[:50]}"


def main():
    print("=" * 60)
    print("Filtering bad images from training data")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BAD_DIR, exist_ok=True)

    png_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.png")))
    print(f"Found {len(png_files)} images to check")
    print()

    good_count = 0
    bad_count = 0
    bad_reasons = {}

    for i, img_path in enumerate(png_files):
        if i % 1000 == 0:
            print(f"Checking image {i}/{len(png_files)}... (good: {good_count}, bad: {bad_count})")

        ok, reason = test_image_dewarp(img_path)

        basename = os.path.basename(img_path)
        gt_name = basename.replace('.png', '.gt.txt')
        gt_path = os.path.join(SOURCE_DIR, gt_name)

        if ok:
            # Copy good files to filtered directory
            shutil.copy2(img_path, os.path.join(OUTPUT_DIR, basename))
            if os.path.exists(gt_path):
                shutil.copy2(gt_path, os.path.join(OUTPUT_DIR, gt_name))
            good_count += 1
        else:
            # Move bad files to bad directory
            shutil.copy2(img_path, os.path.join(BAD_DIR, basename))
            if os.path.exists(gt_path):
                shutil.copy2(gt_path, os.path.join(BAD_DIR, gt_name))
            bad_count += 1
            bad_reasons[reason] = bad_reasons.get(reason, 0) + 1

    print()
    print("=" * 60)
    print(f"Results:")
    print(f"  Good images: {good_count}")
    print(f"  Bad images: {bad_count}")
    print(f"  Output directory: {OUTPUT_DIR}")

    if bad_reasons:
        print()
        print("Bad image reasons:")
        for reason, count in sorted(bad_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print()
    print("To train on filtered data, use:")
    print(f'  "training_data_lines/public_line_images_filtered/*.png"')


if __name__ == "__main__":
    main()
