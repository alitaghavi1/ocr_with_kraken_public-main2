"""
Build context model from a corpus of Persian books.

This script processes text files from your library of old Persian books
to build better bigram statistics for OCR post-processing.

Supports:
- Plain text files (.txt)
- Can recursively scan directories
- Handles various Persian text encodings
"""

import re
import sys
import io
from pathlib import Path
from collections import Counter, defaultdict
import pickle

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent


class CorpusProcessor:
    def __init__(self):
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)
        self.trigrams = defaultdict(Counter)  # For better context
        self.total_words = 0
        self.total_bigrams = 0
        self.total_trigrams = 0
        self.files_processed = 0

    def normalize_word(self, word):
        """Normalize Persian word for matching."""
        if not word:
            return None

        # Remove edge punctuation but keep Persian/Arabic characters
        word = re.sub(r'^[^\w\u0600-\u06FF\u0750-\u077F]+', '', word)
        word = re.sub(r'[^\w\u0600-\u06FF\u0750-\u077F]+$', '', word)

        # Skip if too short or only numbers
        if len(word) < 2 or word.isdigit():
            return None

        return word

    def process_text(self, text):
        """Process a text and extract n-grams."""
        # Split into words
        words = text.split()

        # Normalize words
        normalized = []
        for w in words:
            norm = self.normalize_word(w)
            if norm:
                normalized.append(norm)
                self.word_freq[norm] += 1
                self.total_words += 1

        # Extract bigrams
        for i in range(len(normalized) - 1):
            w1, w2 = normalized[i], normalized[i + 1]
            self.bigrams[w1][w2] += 1
            self.total_bigrams += 1

        # Extract trigrams (for even better context)
        for i in range(len(normalized) - 2):
            w1, w2, w3 = normalized[i], normalized[i + 1], normalized[i + 2]
            key = f"{w1}|{w2}"
            self.trigrams[key][w3] += 1
            self.total_trigrams += 1

    def process_file(self, file_path):
        """Process a single text file."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']:
                try:
                    text = file_path.read_text(encoding=encoding)
                    self.process_text(text)
                    self.files_processed += 1
                    return True
                except UnicodeDecodeError:
                    continue
            return False
        except Exception as e:
            return False

    def process_directory(self, dir_path, pattern="*.txt", recursive=True):
        """Process all text files in a directory."""
        dir_path = Path(dir_path)

        if not dir_path.exists():
            print(f"Directory not found: {dir_path}")
            return

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        print(f"Found {len(files)} text files in {dir_path}")

        for i, f in enumerate(files):
            self.process_file(f)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(files)} files...")

        print(f"  Completed: {self.files_processed} files processed")

    def get_stats(self):
        """Return corpus statistics."""
        return {
            'files': self.files_processed,
            'total_words': self.total_words,
            'unique_words': len(self.word_freq),
            'total_bigrams': self.total_bigrams,
            'unique_bigram_pairs': len(self.bigrams),
            'total_trigrams': self.total_trigrams,
        }

    def save_model(self, path):
        """Save the corpus model."""
        path = Path(path)
        data = {
            'word_freq': dict(self.word_freq),
            'bigrams': {k: dict(v) for k, v in self.bigrams.items()},
            'trigrams': {k: dict(v) for k, v in self.trigrams.items()},
            'total_words': self.total_words,
            'total_bigrams': self.total_bigrams,
            'total_trigrams': self.total_trigrams,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved corpus model to: {path}")

    def save_vocabulary(self, path, min_freq=2):
        """Save unique vocabulary to text file."""
        path = Path(path)

        # Filter by frequency
        vocab = [(w, f) for w, f in self.word_freq.items() if f >= min_freq]
        vocab.sort(key=lambda x: -x[1])  # Sort by frequency

        with open(path, 'w', encoding='utf-8') as f:
            for word, freq in vocab:
                f.write(f"{word}\n")

        print(f"Saved {len(vocab)} words (freq >= {min_freq}) to: {path}")

    def merge_with_existing(self, existing_model_path):
        """Merge with an existing model."""
        path = Path(existing_model_path)
        if not path.exists():
            print(f"No existing model found at: {path}")
            return

        print(f"Loading existing model: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Merge word frequencies
        for word, freq in data.get('word_freq', {}).items():
            self.word_freq[word] += freq
            self.total_words += freq

        # Merge bigrams
        for w1, following in data.get('bigrams', {}).items():
            for w2, count in following.items():
                self.bigrams[w1][w2] += count
                self.total_bigrams += count

        # Merge trigrams if present
        for key, following in data.get('trigrams', {}).items():
            for w3, count in following.items():
                self.trigrams[key][w3] += count
                self.total_trigrams += count

        print("  Merged successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build context model from Persian book corpus"
    )
    parser.add_argument("corpus_dir", type=str, nargs='?',
                        help="Directory containing Persian text files")
    parser.add_argument("--pattern", type=str, default="*.txt",
                        help="File pattern to match (default: *.txt)")
    parser.add_argument("--no-recursive", action="store_true",
                        help="Don't search subdirectories")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing training model")
    parser.add_argument("--output", "-o", type=str,
                        help="Output model path")
    parser.add_argument("--save-vocab", action="store_true",
                        help="Also save vocabulary list")
    parser.add_argument("--min-freq", type=int, default=2,
                        help="Minimum frequency for vocabulary (default: 2)")

    args = parser.parse_args()

    if not args.corpus_dir:
        print("=" * 60)
        print("Build Context Model from Persian Book Corpus")
        print("=" * 60)
        print("\nUsage:")
        print("  python build_corpus_model.py <corpus_directory>")
        print("\nOptions:")
        print("  --pattern *.txt    File pattern to match")
        print("  --no-recursive     Don't search subdirectories")
        print("  --merge            Merge with existing training model")
        print("  --save-vocab       Save vocabulary list")
        print("  --min-freq N       Min frequency for vocab (default: 2)")
        print("\nExamples:")
        print('  python build_corpus_model.py "D:\\Persian Books"')
        print('  python build_corpus_model.py "D:\\Persian Books" --merge --save-vocab')
        return

    print("=" * 60)
    print("Building Context Model from Corpus")
    print("=" * 60)

    processor = CorpusProcessor()

    # Optionally merge with existing model first
    if args.merge:
        existing = BASE_DIR / "ocr_context_model.pkl"
        processor.merge_with_existing(existing)

    # Process corpus
    print(f"\nProcessing corpus: {args.corpus_dir}")
    processor.process_directory(
        args.corpus_dir,
        pattern=args.pattern,
        recursive=not args.no_recursive
    )

    # Show stats
    stats = processor.get_stats()
    print("\n" + "=" * 60)
    print("Corpus Statistics")
    print("=" * 60)
    print(f"  Files processed:    {stats['files']:,}")
    print(f"  Total words:        {stats['total_words']:,}")
    print(f"  Unique words:       {stats['unique_words']:,}")
    print(f"  Total bigrams:      {stats['total_bigrams']:,}")
    print(f"  Unique bigram pairs:{stats['unique_bigram_pairs']:,}")
    print(f"  Total trigrams:     {stats['total_trigrams']:,}")

    # Save model
    output_path = args.output or (BASE_DIR / "ocr_context_model.pkl")
    processor.save_model(output_path)

    # Optionally save vocabulary
    if args.save_vocab:
        vocab_path = BASE_DIR / "corpus_vocabulary.txt"
        processor.save_vocabulary(vocab_path, min_freq=args.min_freq)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nTo use the new model for OCR post-processing:")
    print(f"  python post_process_context.py -i your_ocr.txt -o corrected.txt")


if __name__ == "__main__":
    main()
