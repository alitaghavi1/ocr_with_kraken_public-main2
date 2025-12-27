"""
Download Concordia Persian Handwritten Database sample.
"""

import urllib.request
from pathlib import Path
import ssl

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "training_data_lines" / "concordia_persian"

# Sample database URL
SAMPLE_URL = "http://users.encs.concordia.ca/~j_sadri/Sample_DB.rar"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dest = OUTPUT_DIR / "Sample_DB.rar"

    if dest.exists():
        print(f"File already exists: {dest}")
        print(f"Size: {dest.stat().st_size / 1024:.1f} KB")
    else:
        print(f"Downloading from: {SAMPLE_URL}")
        print("This may take a moment...")

        try:
            # Create SSL context that doesn't verify (some older sites have cert issues)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            urllib.request.urlretrieve(SAMPLE_URL, dest)
            print(f"Downloaded: {dest}")
            print(f"Size: {dest.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"Error: {e}")
            print("\nTry downloading manually from:")
            print(f"  {SAMPLE_URL}")
            return

    print(f"\nNext steps:")
    print(f"1. Extract the RAR file to: {OUTPUT_DIR}")
    print(f"2. Explore the contents to see if it has text lines + transcriptions")
    print(f"\nNote: This is just a SAMPLE. Full database requires contacting:")
    print(f"  Dr. Javad Sadri: j_sadri@encs.concordia.ca")

if __name__ == "__main__":
    main()
