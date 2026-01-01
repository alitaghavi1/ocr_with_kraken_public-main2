"""
Build Bigram and Trigram Context Model from Ganjoor Persian Texts.

This script processes all Ganjoor text files to create a statistical language model
for use in OCR post-processing. The model captures:
- Word frequencies
- Bigram (word pair) frequencies
- Trigram (word triple) frequencies

Usage:
    python build_ganjoor_context_model.py

Output:
    ocr_context_model.pkl - Pickled context model for post-processing
"""

import re
import sys
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).parent
GANJOOR_DIR = BASE_DIR / "ganjoor_texts"
OUTPUT_FILE = BASE_DIR / "ocr_context_model.pkl"


def normalize_word(word):
    """
    Normalize Persian/Arabic word for context model.

    - Remove non-Persian punctuation from edges
    - Keep Persian diacritics and characters
    - Return None for very short words
    """
    if not word:
        return None

    # Remove edge punctuation but keep Persian/Arabic characters
    word = re.sub(r'^[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+', '', word)
    word = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+$', '', word)

    # Must be at least 2 characters
    return word if len(word) >= 2 else None


def extract_words(text):
    """Extract and normalize words from text."""
    # Split on whitespace and common separators
    raw_words = re.split(r'[\s،؛:.!؟\n\r\t]+', text)

    words = []
    for w in raw_words:
        normalized = normalize_word(w)
        if normalized:
            words.append(normalized)

    return words


def build_context_model(text_files):
    """
    Build bigram and trigram model from text files.

    Returns dict with:
    - word_freq: Counter of word frequencies
    - bigrams: dict[word1] -> Counter[word2]
    - trigrams: dict["w1|w2"] -> Counter[w3]
    - total_bigrams: int
    - total_trigrams: int
    """
    word_freq = Counter()
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    total_bigrams = 0
    total_trigrams = 0
    total_words = 0

    for file_path in text_files:
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore')
            words = extract_words(text)

            # Update word frequencies
            word_freq.update(words)
            total_words += len(words)

            # Extract bigrams
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                bigrams[w1][w2] += 1
                total_bigrams += 1

            # Extract trigrams
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                trigram_key = f"{w1}|{w2}"
                trigrams[trigram_key][w3] += 1
                total_trigrams += 1

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    return {
        'word_freq': dict(word_freq),
        'bigrams': {k: dict(v) for k, v in bigrams.items()},
        'trigrams': {k: dict(v) for k, v in trigrams.items()},
        'total_bigrams': total_bigrams,
        'total_trigrams': total_trigrams,
        'total_words': total_words,
        'unique_words': len(word_freq)
    }


def show_sample_ngrams(model, n=10):
    """Show sample bigrams and trigrams for verification."""
    print(f"\n{'='*60}")
    print("Sample Bigrams (most common word pairs):")
    print('='*60)

    # Find most common bigrams
    all_bigrams = []
    for w1, following in model['bigrams'].items():
        for w2, count in following.items():
            all_bigrams.append((w1, w2, count))

    all_bigrams.sort(key=lambda x: -x[2])

    for w1, w2, count in all_bigrams[:n]:
        print(f"  {w1} -> {w2}: {count}")

    print(f"\n{'='*60}")
    print("Sample Trigrams (most common word triples):")
    print('='*60)

    # Find most common trigrams
    all_trigrams = []
    for key, following in model['trigrams'].items():
        w1, w2 = key.split('|')
        for w3, count in following.items():
            all_trigrams.append((w1, w2, w3, count))

    all_trigrams.sort(key=lambda x: -x[3])

    for w1, w2, w3, count in all_trigrams[:n]:
        print(f"  {w1} {w2} -> {w3}: {count}")


def show_context_examples(model):
    """Show examples of how context works for common words."""
    print(f"\n{'='*60}")
    print("Context Examples:")
    print('='*60)

    test_words = ['در', 'از', 'که', 'را', 'این', 'با', 'بر', 'چون', 'نی', 'دل']

    for word in test_words:
        if word in model['bigrams']:
            following = model['bigrams'][word]
            top_following = sorted(following.items(), key=lambda x: -x[1])[:5]

            print(f"\n  After '{word}':")
            for next_word, count in top_following:
                print(f"    -> {next_word} ({count})")


def main():
    print("="*60)
    print("Building Bigram/Trigram Context Model from Ganjoor Texts")
    print("="*60)
    print()

    # Check for Ganjoor texts
    if not GANJOOR_DIR.exists():
        print(f"ERROR: Ganjoor texts directory not found: {GANJOOR_DIR}")
        print("Run: python download_ganjoor.py")
        return

    # Find all text files
    text_files = list(GANJOOR_DIR.glob("*.txt"))
    print(f"Found {len(text_files)} text files in {GANJOOR_DIR}")

    if not text_files:
        print("No text files found!")
        return

    # Build model
    print("\nBuilding context model...")
    print("  Processing files...")

    model = build_context_model(text_files)

    # Show statistics
    print(f"\n{'='*60}")
    print("Context Model Statistics:")
    print('='*60)
    print(f"  Total words processed: {model['total_words']:,}")
    print(f"  Unique words: {model['unique_words']:,}")
    print(f"  Total bigrams: {model['total_bigrams']:,}")
    print(f"  Unique bigram contexts: {len(model['bigrams']):,}")
    print(f"  Total trigrams: {model['total_trigrams']:,}")
    print(f"  Unique trigram contexts: {len(model['trigrams']):,}")

    # Show samples
    show_sample_ngrams(model)
    show_context_examples(model)

    # Save model
    print(f"\n{'='*60}")
    print(f"Saving model to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(model, f)

    # Verify file size
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    print(f"\n{'='*60}")
    print("Complete!")
    print('='*60)
    print("\nThe context model is now ready for use with post-processing.")
    print("It will be automatically loaded by EnhancedPostProcessor.")


if __name__ == "__main__":
    main()
