"""
Post-process OCR output using fuzzy matching against a word dictionary.

This corrects OCR errors by matching words against known vocabulary.
"""

import re
import sys
from pathlib import Path
from collections import Counter

# Try to import fuzzy matching library
try:
    from rapidfuzz import fuzz, process
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        FUZZY_LIB = None
        print("WARNING: No fuzzy matching library found.")
        print("Install with: pip install rapidfuzz")


class OCRPostProcessor:
    def __init__(self, dictionary_path=None, min_word_length=2,
                 fuzzy_threshold=80, max_candidates=5):
        """
        Initialize the post-processor.

        Args:
            dictionary_path: Path to word list file (one word per line)
            min_word_length: Minimum word length to attempt correction
            fuzzy_threshold: Minimum similarity score (0-100) to accept match
            max_candidates: Maximum candidates to consider for each word
        """
        self.min_word_length = min_word_length
        self.fuzzy_threshold = fuzzy_threshold
        self.max_candidates = max_candidates
        self.dictionary = set()
        self.word_freq = Counter()

        if dictionary_path:
            self.load_dictionary(dictionary_path)

    def load_dictionary(self, path):
        """Load word dictionary from file."""
        path = Path(path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.dictionary.add(word)
            print(f"Loaded {len(self.dictionary)} words from dictionary")

    def add_words(self, words):
        """Add words to dictionary."""
        for word in words:
            word = word.strip()
            if word:
                self.dictionary.add(word)
                self.word_freq[word] += 1

    def build_from_training_data(self, training_dir, pattern="*.gt.txt"):
        """Build dictionary from training ground truth files."""
        training_dir = Path(training_dir)
        if not training_dir.exists():
            print(f"Training directory not found: {training_dir}")
            return

        print(f"Building dictionary from: {training_dir}")
        word_count = 0

        for gt_file in training_dir.glob(pattern):
            try:
                text = gt_file.read_text(encoding='utf-8').strip()
                words = text.split()
                for word in words:
                    # Clean word (remove punctuation at edges)
                    word = re.sub(r'^[^\w\u0600-\u06FF]+|[^\w\u0600-\u06FF]+$', '', word)
                    if word and len(word) >= self.min_word_length:
                        self.dictionary.add(word)
                        self.word_freq[word] += 1
                        word_count += 1
            except:
                pass

        print(f"  Extracted {len(self.dictionary)} unique words from {word_count} occurrences")

    def find_best_match(self, word):
        """Find best matching word from dictionary using fuzzy matching."""
        if not FUZZY_LIB:
            return word, 0

        if not self.dictionary:
            return word, 0

        # Skip if word is already in dictionary
        if word in self.dictionary:
            return word, 100

        # Skip short words
        if len(word) < self.min_word_length:
            return word, 0

        # Find best matches
        matches = process.extract(
            word,
            self.dictionary,
            scorer=fuzz.ratio,
            limit=self.max_candidates
        )

        if not matches:
            return word, 0

        # Get best match
        best_match, score, *_ = matches[0]

        # Apply frequency bonus (prefer common words)
        if self.word_freq[best_match] > 10:
            score = min(100, score + 5)

        return best_match, score

    def correct_word(self, word):
        """Correct a single word if possible."""
        if len(word) < self.min_word_length:
            return word, False

        best_match, score = self.find_best_match(word)

        if score >= self.fuzzy_threshold and best_match != word:
            return best_match, True

        return word, False

    def process_text(self, text, verbose=False):
        """
        Process OCR text and correct errors.

        Returns:
            (corrected_text, corrections_made)
        """
        words = text.split()
        corrected_words = []
        corrections = []

        for word in words:
            corrected, was_corrected = self.correct_word(word)
            corrected_words.append(corrected)

            if was_corrected:
                corrections.append((word, corrected))
                if verbose:
                    print(f"  '{word}' → '{corrected}'")

        corrected_text = ' '.join(corrected_words)
        return corrected_text, corrections

    def save_dictionary(self, path):
        """Save dictionary to file."""
        path = Path(path)
        # Sort by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: -x[1])

        with open(path, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                f.write(f"{word}\n")

        print(f"Saved {len(sorted_words)} words to {path}")


def build_dictionary_from_training():
    """Build and save dictionary from all training data."""
    base_dir = Path(__file__).parent

    processor = OCRPostProcessor(min_word_length=2)

    # Build from all training sources
    sources = [
        base_dir / "training_data_lines" / "balanced_training",
        base_dir / "training_data_words",
        base_dir / "handwritten_training_data",
        base_dir / "combined_training",
    ]

    for source in sources:
        if source.exists():
            processor.build_from_training_data(source)

    # Save dictionary
    dict_path = base_dir / "ocr_dictionary.txt"
    processor.save_dictionary(dict_path)

    # Show top words
    print("\nTop 30 most frequent words:")
    for word, freq in processor.word_freq.most_common(30):
        print(f"  {word}: {freq}")

    return dict_path


def process_ocr_output(text, dictionary_path=None, threshold=80, verbose=True):
    """
    Main function to post-process OCR output.

    Args:
        text: Raw OCR output text
        dictionary_path: Path to word dictionary
        threshold: Fuzzy matching threshold (0-100)
        verbose: Print corrections

    Returns:
        Corrected text
    """
    base_dir = Path(__file__).parent

    # Use default dictionary if not specified
    if dictionary_path is None:
        dictionary_path = base_dir / "ocr_dictionary.txt"

    # Build dictionary if it doesn't exist
    if not Path(dictionary_path).exists():
        print("Dictionary not found. Building from training data...")
        dictionary_path = build_dictionary_from_training()

    # Create processor
    processor = OCRPostProcessor(
        dictionary_path=dictionary_path,
        fuzzy_threshold=threshold
    )

    if verbose:
        print(f"\nProcessing text ({len(text.split())} words)...")
        print("Corrections:")

    corrected, corrections = processor.process_text(text, verbose=verbose)

    if verbose:
        print(f"\nTotal corrections: {len(corrections)}")

    return corrected


# Example usage and CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process OCR output with fuzzy matching")
    parser.add_argument("--build-dict", action="store_true",
                        help="Build dictionary from training data")
    parser.add_argument("--input", "-i", type=str,
                        help="Input file with OCR text")
    parser.add_argument("--text", "-t", type=str,
                        help="Direct text input")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for corrected text")
    parser.add_argument("--threshold", type=int, default=80,
                        help="Fuzzy matching threshold (0-100)")
    parser.add_argument("--dict", "-d", type=str,
                        help="Path to dictionary file")

    args = parser.parse_args()

    if args.build_dict:
        build_dictionary_from_training()
        sys.exit(0)

    # Get input text
    if args.input:
        text = Path(args.input).read_text(encoding='utf-8')
    elif args.text:
        text = args.text
    else:
        print("Usage examples:")
        print("  Build dictionary:  python post_process_ocr.py --build-dict")
        print("  Process file:      python post_process_ocr.py -i ocr_output.txt -o corrected.txt")
        print("  Process text:      python post_process_ocr.py -t 'your text here'")
        sys.exit(0)

    # Process
    corrected = process_ocr_output(
        text,
        dictionary_path=args.dict,
        threshold=args.threshold
    )

    # Output
    print("\n" + "="*50)
    print("ORIGINAL:")
    print(text)
    print("\nCORRECTED:")
    print(corrected)

    if args.output:
        Path(args.output).write_text(corrected, encoding='utf-8')
        print(f"\nSaved to: {args.output}")
