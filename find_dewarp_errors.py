"""
Find images that cause dewarping errors by actually running the dewarp function.
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import kraken's dewarping
try:
    from kraken.lib.lineest import CenterNormalizer
except ImportError:
    print("Error: kraken not installed")
    sys.exit(1)


def test_dewarp(img_path):
    """Test if an image can be dewarped without errors."""
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        arr = np.array(img)

        # Skip very small images
        if arr.shape[0] < 10 or arr.shape[1] < 10:
            return img_path, "Too small"

        # Skip blank images
        if arr.min() == arr.max():
            return img_path, "Blank image"

        # Try the actual dewarp
        normalizer = CenterNormalizer()
        try:
            # This is what kraken does internally
            normalized = normalizer.normalize(arr, cval=np.amax(arr))
        except Exception as e:
            return img_path, f"Dewarp error: {str(e)[:100]}"

        return None, None

    except Exception as e:
        return img_path, f"Error: {str(e)[:100]}"


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
    print(f"Testing dewarping on {len(images)} images in {data_dir}...")
    print("This may take a while...")

    bad_images = []

    # Test images (use single process to avoid multiprocessing issues with kraken)
    for i, img_path in enumerate(images):
        if (i + 1) % 500 == 0:
            print(f"  Tested {i + 1}/{len(images)}... ({len(bad_images)} errors found)")

        path, error = test_dewarp(img_path)
        if path:
            bad_images.append((path, error))

    # Report findings
    print(f"\n{'='*60}")
    print(f"Found {len(bad_images)} images with dewarping errors:")
    print(f"{'='*60}")

    for path, error in bad_images[:30]:  # Show first 30
        print(f"\n{path.name}:")
        print(f"  {error}")

    if len(bad_images) > 30:
        print(f"\n... and {len(bad_images) - 30} more")

    # Save list to file
    if bad_images:
        with open("dewarp_errors.txt", "w") as f:
            for path, error in bad_images:
                f.write(f"{path}\t{error}\n")
        print(f"\nSaved full list to dewarp_errors.txt")

        print(f"\n{'='*60}")
        if '--remove' in sys.argv:
            removed = 0
            for path, _ in bad_images:
                try:
                    path.unlink()
                    gt_path = path.with_suffix('.gt.txt')
                    if gt_path.exists():
                        gt_path.unlink()
                    removed += 1
                except Exception as e:
                    print(f"Error removing {path}: {e}")
            print(f"Removed {removed} images and their ground truth files.")
        else:
            print(f"To remove these images, run:")
            print(f"  python find_dewarp_errors.py {data_dir} --remove")
    else:
        print("\nNo dewarping errors found!")


if __name__ == "__main__":
    main()
