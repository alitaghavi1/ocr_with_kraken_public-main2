"""
Find and optionally remove images that cause dewarping errors in Kraken.
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_image(img_path):
    """Check if an image will cause dewarping issues."""
    try:
        img = Image.open(img_path)
        arr = np.array(img)

        # Check for problematic conditions
        issues = []

        # 1. Check if image is too small
        if arr.shape[0] < 10 or arr.shape[1] < 10:
            issues.append(f"Too small: {arr.shape}")

        # 2. Check if image is completely blank (all same value)
        if arr.min() == arr.max():
            issues.append("Blank image (uniform color)")

        # 3. Check for very thin images (height < 5)
        if arr.shape[0] < 5:
            issues.append(f"Too thin: height={arr.shape[0]}")

        # 4. Check for extremely wide aspect ratio
        if arr.shape[1] > arr.shape[0] * 100:
            issues.append(f"Extreme aspect ratio: {arr.shape[1]/arr.shape[0]:.0f}:1")

        # 5. Check if grayscale or RGB
        if len(arr.shape) == 3 and arr.shape[2] == 4:
            # RGBA - check for fully transparent
            if arr[:,:,3].max() == 0:
                issues.append("Fully transparent image")

        # 6. Check for NaN or Inf values (corrupted)
        if np.isnan(arr).any() or np.isinf(arr).any():
            issues.append("Contains NaN or Inf values")

        if issues:
            return img_path, issues
        return None, None

    except Exception as e:
        return img_path, [f"Error opening: {e}"]

def main():
    if len(sys.argv) < 2:
        data_dir = Path("training_data_lines/balanced_training")
    else:
        data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        sys.exit(1)

    # Find all images
    images = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    print(f"Checking {len(images)} images in {data_dir}...")

    bad_images = []

    # Check images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_image, img): img for img in images}

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 1000 == 0:
                print(f"  Checked {i + 1}/{len(images)}...")

            path, issues = future.result()
            if path:
                bad_images.append((path, issues))

    # Report findings
    print(f"\n{'='*60}")
    print(f"Found {len(bad_images)} problematic images:")
    print(f"{'='*60}")

    for path, issues in bad_images[:50]:  # Show first 50
        print(f"\n{path.name}:")
        for issue in issues:
            print(f"  - {issue}")

    if len(bad_images) > 50:
        print(f"\n... and {len(bad_images) - 50} more")

    # Ask to remove
    if bad_images:
        print(f"\n{'='*60}")
        response = input(f"Remove {len(bad_images)} bad images? (y/n): ").strip().lower()

        if response == 'y':
            removed = 0
            for path, _ in bad_images:
                try:
                    # Remove image
                    path.unlink()
                    # Remove corresponding .gt.txt if exists
                    gt_path = path.with_suffix('.gt.txt')
                    if gt_path.exists():
                        gt_path.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Error removing {path}: {e}")

            print(f"Removed {removed} images and their ground truth files.")
        else:
            # Save list to file
            with open("bad_images.txt", "w") as f:
                for path, issues in bad_images:
                    f.write(f"{path}\t{'; '.join(issues)}\n")
            print("Saved list to bad_images.txt")
    else:
        print("\nNo problematic images found!")

if __name__ == "__main__":
    main()
