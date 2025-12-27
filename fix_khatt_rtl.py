"""
Fix KHATT ground truth files - reverse LTR text back to RTL.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent
KHATT_DIR = BASE_DIR / "training_data_lines" / "khatt_lines"

def main():
    gt_files = list(KHATT_DIR.glob("*.gt.txt"))
    print(f"Found {len(gt_files)} ground truth files")

    fixed = 0
    for gt_file in gt_files:
        try:
            text = gt_file.read_text(encoding='utf-8').strip()

            # Reverse the text (LTR -> RTL)
            reversed_text = text[::-1]

            gt_file.write_text(reversed_text, encoding='utf-8')
            fixed += 1

            if fixed % 500 == 0:
                print(f"  Fixed {fixed} files...")

        except Exception as e:
            print(f"  Error with {gt_file.name}: {e}")

    print(f"\nDone! Fixed {fixed} files.")

    # Show sample
    sample_files = list(KHATT_DIR.glob("*.gt.txt"))[:3]
    print("\nSample (first 3 files):")
    for f in sample_files:
        text = f.read_text(encoding='utf-8').strip()
        print(f"  {f.name}: {text[:50]}...")

if __name__ == "__main__":
    main()
