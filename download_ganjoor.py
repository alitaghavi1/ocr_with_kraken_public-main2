"""
Download Persian classical literature from Ganjoor.

Downloads EPUB files and extracts plain text for corpus building.
Source: https://epub.ganjoor.net/
"""

import urllib.request
import zipfile
import re
import sys
import io
import os
from pathlib import Path
from html.parser import HTMLParser
import time

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent
EPUB_DIR = BASE_DIR / "ganjoor_epub"
TEXT_DIR = BASE_DIR / "ganjoor_texts"

# List of all Ganjoor poets/works (filename without .epub)
GANJOOR_WORKS = [
    "ابن-حسام-خوسفی", "ابن-یمین", "ابوالحسن-ورزی", "ابوالفرج-رونی", "ابوسعید-ابوالخیر",
    "ابوطالب-کلیم", "اثیر-اخسیکتی", "احتشام-الدوله", "احمد-شاملو", "احمد-محمود-امپراطور",
    "اخوان-ثالث", "ادیب-الممالک-فراهانی", "ادیب-صابر", "اسدی-توسی", "اشرف‌الدین-گیلانی",
    "اشرف-مراغه‌ای", "افشین-یداللهی", "امیر-معزی", "امیر-هوشنگ-ابتهاج-سایه",
    "امیرخسرو-دهلوی", "انوری", "اهلی-شیرازی", "اوحدالدین-کرمانی", "اوحدی-مراغه‌ای",
    "ایرج-میرزا", "باباافضل-کاشانی", "باباطاهر", "بابافغانی-شیرازی", "بحرالدین-محمد",
    "بردیا-علومی", "بیدل-دهلوی", "پروین-اعتصامی", "جامی", "جلال-عضد",
    "جلال‌الدین-همایی", "جمال‌الدین-اصفهانی", "جویای-تبریزی", "جیحون-یزدی",
    "حاجب-شیرازی", "حافظ", "حبیب-یغمایی", "حزین-لاهیجی", "حسام-خوسفی",
    "حسرت-همدانی", "حسن-غزنوی", "حسین-منزوی", "حسین‌قلی-مستان", "حکیم-زلالی-خوانساری",
    "حکیم-سبزواری", "خاقانی", "خالد-نقشبندی", "خواجو-کرمانی", "خیالی-بخارایی",
    "خیام", "داراب-افسر-بختیاری", "دقیقی", "رشید-یاسمی", "رشیدالدین-وطواط",
    "رضاقلی-هدایت", "رفیق-اصفهانی", "رهی-معیری", "رودکی", "سحاب-اصفهانی",
    "سراج-قمری", "سرمد-کاشانی", "سعدی", "سعید-حمیدیان", "سلطان-ولد",
    "سلمان-ساوجی", "سنایی", "سهراب-سپهری", "سیدای-نسفی", "سیدحسن-حسینی",
    "سیدمحمد-صفایی", "سیف-فرغانی", "شاپور-نطنزی", "شاطر-عباس-صبوحی",
    "شاه-نعمت‌الله-ولی", "شرف‌الدین-شفایی", "شفق-اصفهانی", "شکیب-اصفهانی",
    "شمس-تبریزی", "شهرام-شیدایی", "شهریار", "شیخ-بهایی", "صائب-تبریزی",
    "صبا-کاشانی", "صباحی-بیدگلی", "صدری-افشار", "صفای-اصفهانی", "طاهره-قرةالعین",
    "طالب-آملی", "طبیب-اصفهانی", "طغرای-مشهدی", "ظهیر-فاریابی", "ظهیری-سمرقندی",
    "عارف-قزوینی", "عبدالحق-دهلوی", "عبدالرحمن-جامی", "عبدالقادر-گیلانی",
    "عبدالله-هاتفی", "عبدالواسع-جبلی", "عبید-زاکانی", "عثمان-مختاری",
    "عراقی", "عرفی-شیرازی", "عطار", "عماد-خراسانی", "عمان-سامانی",
    "عمعق-بخارایی", "عنصرالمعالی", "عنصری", "عیوقی", "غالب-دهلوی",
    "غزالی-مشهدی", "غلامرضا-روحانی", "فتحعلی-خان-صبا", "فخرالدین-اسعد",
    "فرخی-سیستانی", "فرخی-یزدی", "فردوسی", "فروغ-فرخزاد", "فروغی-بسطامی",
    "فصیح-الملک-شیرازی", "فصیحی-هروی", "فضل‌الله-گروسی", "فضولی-بغدادی",
    "فغانی-شیرازی", "فنایی", "فیض-کاشانی", "قاآنی", "قاسم-انوار",
    "قدسی-مشهدی", "قطران-تبریزی", "کسایی-مروزی", "کمال-خجندی",
    "کمال‌الدین-اسماعیل", "لامعی-گرگانی", "لسان‌الغیب-شیرازی", "مجد-همگر",
    "مجدالدین-بغدادی", "مجیرالدین-بیلقانی", "محتشم-کاشانی", "محمد-بن-وصیف",
    "محمدتقی-بهار", "مختاری-غزنوی", "مسرور-تبریزی", "مسعود-سعد-سلمان",
    "مشتاق-اصفهانی", "مشفق-کاشانی", "مشیری", "معزی", "مفتون-همدانی",
    "ملا-هادی-سبزواری", "ملاقلی-قاری", "ملک‌الشعرا-بهار", "منوچهر-آتشی",
    "منوچهری", "مولوی", "میبدی", "میرزا-آقاخان-کرمانی", "میرزاده-عشقی",
    "ناصرخسرو", "نثار-احمد-جان", "نزاری-قهستانی", "نشاط-اصفهانی", "نظامی",
    "نظام‌الدین-استرآبادی", "نظیری-نیشابوری", "نقی-آتشی", "نواب-صفا",
    "نویسنده-زن-اقبال-آذر", "نیلوفر-لاریجانی", "نیما-یوشیج", "واعظ-قزوینی",
    "وحدت-کرمانشاهی", "وحشی-بافقی", "وصال-شیرازی", "وفایی-شوشتری",
    "وقار-شیرازی", "هاتف-اصفهانی", "هجویری", "هدایت", "هلالی-جغتایی",
    "همام-تبریزی", "همای-شیرازی", "یحیی-کاشی"
]

# Or download a smaller subset of most important classical works
ESSENTIAL_WORKS = [
    "فردوسی",      # Shahnameh - Epic of Kings
    "حافظ",        # Hafez - Ghazals
    "سعدی",        # Saadi - Gulistan, Bustan
    "مولوی",       # Rumi - Masnavi
    "عطار",        # Attar - Conference of Birds
    "خیام",        # Khayyam - Rubaiyat
    "نظامی",       # Nizami - Khamsa
    "خاقانی",      # Khaghani
    "سنایی",       # Sanai
    "ناصرخسرو",    # Naser Khosrow
    "رودکی",       # Rudaki
    "فرخی-سیستانی", # Farrokhi
    "عنصری",       # Onsori
    "منوچهری",     # Manuchehri
    "جامی",        # Jami
    "انوری",       # Anvari
    "اوحدی-مراغه‌ای", # Owhadi
    "عراقی",       # Iraqi
    "خواجو-کرمانی", # Khwaju
    "صائب-تبریزی", # Saeb
]


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML."""
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'title'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()

    def handle_endtag(self, tag):
        if tag.lower() in {'p', 'div', 'br', 'h1', 'h2', 'h3', 'h4'}:
            self.text_parts.append('\n')
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text_parts.append(text + ' ')

    def get_text(self):
        return ''.join(self.text_parts)


def extract_text_from_epub(epub_path):
    """Extract plain text from EPUB file."""
    text_parts = []

    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith(('.xhtml', '.html', '.htm', '.xml')):
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')

                        # Parse HTML and extract text
                        parser = HTMLTextExtractor()
                        parser.feed(content)
                        text = parser.get_text()

                        # Clean up
                        text = re.sub(r'\s+', ' ', text)
                        text = re.sub(r'\n\s*\n', '\n\n', text)

                        if text.strip():
                            text_parts.append(text)
                    except:
                        pass
    except Exception as e:
        print(f"  Error extracting: {e}")
        return None

    return '\n\n'.join(text_parts)


def download_epub(name, output_dir):
    """Download a single EPUB from Ganjoor."""
    # URL-encode the Persian filename
    from urllib.parse import quote
    encoded_name = quote(name)
    url = f"https://i.ganjoor.net/epub/{encoded_name}.epub"
    output_path = output_dir / f"{name}.epub"

    if output_path.exists():
        return output_path

    try:
        print(f"  Downloading: {name}")
        # Use urlopen with proper headers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        time.sleep(0.3)  # Be nice to the server
        return output_path
    except Exception as e:
        print(f"  Error: {e}")
        return None


def get_epub_links_from_page():
    """Scrape actual EPUB links from Ganjoor page."""
    print("Fetching EPUB links from epub.ganjoor.net...")

    try:
        url = "https://epub.ganjoor.net/"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')

        # Find all epub links - pattern: href="https://i.ganjoor.net/epub/....epub"
        import re
        pattern = r'href="(https://i\.ganjoor\.net/epub/[^"]+\.epub)"'
        links = re.findall(pattern, html)

        print(f"  Found {len(links)} EPUB links")
        return links
    except Exception as e:
        print(f"  Error fetching page: {e}")
        return []


def download_epub_direct(url, output_dir):
    """Download EPUB directly from URL."""
    # Extract filename from URL
    from urllib.parse import unquote
    filename = unquote(url.split('/')[-1])
    output_path = output_dir / filename

    if output_path.exists():
        return output_path

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return output_path
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Ganjoor Persian literature")
    parser.add_argument("--all", action="store_true", help="Download all works")
    parser.add_argument("--essential", action="store_true", help="Download only essential 20 works")
    parser.add_argument("--list", type=str, help="File with list of works to download")
    parser.add_argument("--extract-only", action="store_true", help="Only extract text from existing EPUBs")

    args = parser.parse_args()

    print("=" * 60)
    print("Ganjoor Persian Literature Downloader")
    print("=" * 60)

    # Create directories
    EPUB_DIR.mkdir(exist_ok=True)
    TEXT_DIR.mkdir(exist_ok=True)

    # Get actual EPUB links from page
    if not args.extract_only:
        epub_links = get_epub_links_from_page()

        if not epub_links:
            print("Failed to get EPUB links. Try --extract-only if you have EPUBs.")
            return

        print(f"\nDownloading {len(epub_links)} works to: {EPUB_DIR}")

        downloaded = 0
        for i, url in enumerate(epub_links):
            result = download_epub_direct(url, EPUB_DIR)
            if result:
                downloaded += 1

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(epub_links)} ({downloaded} downloaded)")

            time.sleep(0.2)  # Be nice to server

        print(f"  Downloaded: {downloaded}/{len(epub_links)}")

    # Extract text from all EPUBs
    print(f"\nExtracting text to: {TEXT_DIR}")
    epub_files = list(EPUB_DIR.glob("*.epub"))
    print(f"Found {len(epub_files)} EPUB files")

    total_words = 0
    for i, epub_path in enumerate(epub_files):
        text = extract_text_from_epub(epub_path)
        if text:
            txt_path = TEXT_DIR / f"{epub_path.stem}.txt"
            txt_path.write_text(text, encoding='utf-8')
            word_count = len(text.split())
            total_words += word_count

        if (i + 1) % 20 == 0:
            print(f"  Extracted: {i + 1}/{len(epub_files)}")

    # Summary
    txt_files = list(TEXT_DIR.glob("*.txt"))
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"EPUB files: {len(epub_files)}")
    print(f"Text files: {len(txt_files)}")
    print(f"Total words: ~{total_words:,}")
    print(f"\nText files saved to: {TEXT_DIR}")
    print(f"\nNext step - build context model:")
    print(f'  python build_corpus_model.py "{TEXT_DIR}" --merge --save-vocab')


if __name__ == "__main__":
    main()
