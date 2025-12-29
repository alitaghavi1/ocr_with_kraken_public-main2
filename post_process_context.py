"""
Context-Aware OCR Post-Processor with Fuzzy Matching.

Combines:
1. Fuzzy matching against Persian dictionary
2. Context-based scoring using word co-occurrences (bigrams)

The position of a word in the sentence affects correction:
- Uses left and right context words to score candidates
- Prefers candidates that commonly appear together with neighbors
"""

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import pickle

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


class ContextAwarePostProcessor:
    def __init__(self, dictionary_path=None, min_word_length=2,
                 fuzzy_threshold=75, context_weight=0.3, max_candidates=10):
        """
        Initialize context-aware post-processor.

        Args:
            dictionary_path: Path to Persian word list (one word per line)
            min_word_length: Minimum word length to attempt correction
            fuzzy_threshold: Minimum fuzzy score (0-100) to consider a match
            context_weight: Weight for context score (0-1), higher = more context influence
            max_candidates: Maximum candidates to consider per word
        """
        self.min_word_length = min_word_length
        self.fuzzy_threshold = fuzzy_threshold
        self.context_weight = context_weight
        self.max_candidates = max_candidates

        self.dictionary = set()
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)  # bigrams[word1][word2] = count
        self.trigrams = defaultdict(Counter)  # trigrams["w1|w2"][w3] = count
        self.total_bigrams = 0
        self.total_trigrams = 0

        if dictionary_path:
            self.load_dictionary(dictionary_path)

    def load_dictionary(self, path):
        """Load Persian dictionary from file."""
        path = Path(path)
        if not path.exists():
            print(f"Dictionary not found: {path}")
            return

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if word and len(word) >= self.min_word_length:
                    self.dictionary.add(word)

        print(f"Loaded {len(self.dictionary):,} words from dictionary")

    def build_context_from_corpus(self, corpus_path=None, text_files=None):
        """
        Build bigram statistics from corpus.

        Args:
            corpus_path: Path to a single text file
            text_files: List of paths to text files (e.g., training .gt.txt files)
        """
        print("Building context model (bigrams)...")

        texts = []

        if corpus_path:
            corpus_path = Path(corpus_path)
            if corpus_path.exists():
                texts.append(corpus_path.read_text(encoding='utf-8', errors='ignore'))

        if text_files:
            for tf in text_files:
                try:
                    texts.append(Path(tf).read_text(encoding='utf-8', errors='ignore'))
                except:
                    pass

        # Extract bigrams
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                w1 = self._normalize(words[i])
                w2 = self._normalize(words[i + 1])
                if w1 and w2:
                    self.bigrams[w1][w2] += 1
                    self.word_freq[w1] += 1
                    self.total_bigrams += 1

            # Count last word
            if words:
                self.word_freq[self._normalize(words[-1])] += 1

        print(f"  Bigrams: {self.total_bigrams:,}")
        print(f"  Unique words in context: {len(self.word_freq):,}")

    def build_context_from_training(self, training_dir, pattern="*.gt.txt"):
        """Build context model from training ground truth files."""
        training_dir = Path(training_dir)
        if not training_dir.exists():
            print(f"Training directory not found: {training_dir}")
            return

        gt_files = list(training_dir.glob(pattern))
        print(f"Building context from {len(gt_files)} training files...")
        self.build_context_from_corpus(text_files=gt_files)

    def _normalize(self, word):
        """Normalize word for matching."""
        # Remove edge punctuation but keep Persian characters
        word = re.sub(r'^[^\w\u0600-\u06FF]+|[^\w\u0600-\u06FF]+$', '', word)
        return word if len(word) >= self.min_word_length else None

    def get_bigram_score(self, prev_word, word, next_word, prev_prev_word=None):
        """
        Score a word based on how well it fits with neighbors.

        Uses both bigrams and trigrams for better context.
        Returns score between 0-100.
        """
        if self.total_bigrams == 0:
            return 50  # Neutral if no context model

        score = 0
        count = 0

        word_norm = self._normalize(word) if word else None

        # Score based on (prev_word, word) bigram
        if prev_word:
            prev_norm = self._normalize(prev_word)
            if prev_norm and prev_norm in self.bigrams:
                following = self.bigrams[prev_norm]
                if word_norm and word_norm in following:
                    freq = following[word_norm] / max(1, sum(following.values()))
                    score += min(100, freq * 500)
                count += 1

        # Score based on (word, next_word) bigram
        if next_word and word_norm:
            next_norm = self._normalize(next_word)
            if word_norm in self.bigrams:
                following = self.bigrams[word_norm]
                if next_norm and next_norm in following:
                    freq = following[next_norm] / max(1, sum(following.values()))
                    score += min(100, freq * 500)
                count += 1

        # Trigram score: (prev_prev, prev) -> word
        if self.total_trigrams > 0 and prev_prev_word and prev_word:
            prev_prev_norm = self._normalize(prev_prev_word)
            prev_norm = self._normalize(prev_word)
            if prev_prev_norm and prev_norm:
                trigram_key = f"{prev_prev_norm}|{prev_norm}"
                if trigram_key in self.trigrams:
                    following = self.trigrams[trigram_key]
                    if word_norm and word_norm in following:
                        freq = following[word_norm] / max(1, sum(following.values()))
                        # Trigrams are more specific, weight them higher
                        score += min(100, freq * 800)
                        count += 1

        return score / max(1, count) if count > 0 else 50

    def get_candidates(self, word):
        """Get fuzzy match candidates from dictionary."""
        if not FUZZY_LIB or not self.dictionary:
            return [(word, 100)]

        if word in self.dictionary:
            return [(word, 100)]

        if len(word) < self.min_word_length:
            return [(word, 0)]

        # Find fuzzy matches
        matches = process.extract(
            word,
            self.dictionary,
            scorer=fuzz.ratio,
            limit=self.max_candidates
        )

        candidates = []
        for match in matches:
            candidate, score = match[0], match[1]
            if score >= self.fuzzy_threshold:
                candidates.append((candidate, score))

        return candidates if candidates else [(word, 0)]

    def correct_word_with_context(self, word, prev_word, next_word, prev_prev_word=None):
        """
        Correct a word using both fuzzy matching and context.

        Args:
            word: The word to correct
            prev_word: Previous word in sentence (or None)
            next_word: Next word in sentence (or None)
            prev_prev_word: Word before prev_word for trigram context (or None)

        Returns:
            (corrected_word, was_corrected, debug_info)
        """
        if len(word) < self.min_word_length:
            return word, False, None

        # Already in dictionary?
        if word in self.dictionary:
            return word, False, None

        candidates = self.get_candidates(word)

        if not candidates or candidates[0][1] == 0:
            return word, False, None

        # Score each candidate with context
        best_candidate = word
        best_score = 0
        best_info = None

        for candidate, fuzzy_score in candidates:
            # Context score (now with trigram support)
            context_score = self.get_bigram_score(prev_word, candidate, next_word, prev_prev_word)

            # Combined score
            combined = (1 - self.context_weight) * fuzzy_score + self.context_weight * context_score

            if combined > best_score:
                best_score = combined
                best_candidate = candidate
                best_info = {
                    'original': word,
                    'candidate': candidate,
                    'fuzzy': fuzzy_score,
                    'context': context_score,
                    'combined': combined,
                    'prev': prev_word,
                    'next': next_word
                }

        was_corrected = best_candidate != word and best_score >= self.fuzzy_threshold
        return best_candidate, was_corrected, best_info if was_corrected else None

    def process_text(self, text, verbose=False):
        """
        Process OCR text with context-aware correction.

        Returns:
            (corrected_text, list_of_corrections)
        """
        words = text.split()
        corrected_words = []
        corrections = []

        for i, word in enumerate(words):
            prev_prev_word = words[i - 2] if i > 1 else None
            prev_word = words[i - 1] if i > 0 else None
            next_word = words[i + 1] if i < len(words) - 1 else None

            corrected, was_corrected, info = self.correct_word_with_context(
                word, prev_word, next_word, prev_prev_word
            )

            corrected_words.append(corrected)

            if was_corrected:
                corrections.append(info)
                if verbose:
                    try:
                        print(f"  '{word}' -> '{corrected}' "
                              f"(fuzzy:{info['fuzzy']:.0f}, context:{info['context']:.0f})")
                    except:
                        print(f"  [correction made]")

        return ' '.join(corrected_words), corrections

    def save_model(self, path):
        """Save context model to file."""
        path = Path(path)
        data = {
            'bigrams': dict(self.bigrams),
            'word_freq': dict(self.word_freq),
            'total_bigrams': self.total_bigrams
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved context model to: {path}")

    def load_model(self, path):
        """Load context model from file."""
        path = Path(path)
        if not path.exists():
            print(f"Model not found: {path}")
            return False

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in data.get('bigrams', {}).items()})
        self.word_freq = Counter(data.get('word_freq', {}))
        self.total_bigrams = data.get('total_bigrams', 0)

        # Load trigrams if available (from corpus model)
        if 'trigrams' in data:
            self.trigrams = defaultdict(Counter, {k: Counter(v) for k, v in data['trigrams'].items()})
            self.total_trigrams = data.get('total_trigrams', 0)
            print(f"Loaded context model: {self.total_bigrams:,} bigrams, {self.total_trigrams:,} trigrams")
        else:
            print(f"Loaded context model: {self.total_bigrams:,} bigrams")
        return True


def build_full_model():
    """Build dictionary and context model from all sources."""
    base_dir = Path(__file__).parent

    # Check for Persian dictionary
    dict_path = base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt"
    if not dict_path.exists():
        print("Persian dictionary not found!")
        print("Run: python download_persian_dict.py")
        return None

    # Create processor
    processor = ContextAwarePostProcessor(
        dictionary_path=dict_path,
        fuzzy_threshold=75,
        context_weight=0.3
    )

    # Build context from training data
    training_sources = [
        base_dir / "training_data_lines" / "balanced_training",
        base_dir / "training_data_words",
        base_dir / "combined_training",
    ]

    for source in training_sources:
        if source.exists():
            processor.build_context_from_training(source)

    # Save model
    model_path = base_dir / "ocr_context_model.pkl"
    processor.save_model(model_path)

    return processor


def process_ocr_file(input_path, output_path=None, verbose=True):
    """Process an OCR output file."""
    base_dir = Path(__file__).parent

    # Load or build model
    model_path = base_dir / "ocr_context_model.pkl"
    dict_path = base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt"

    processor = ContextAwarePostProcessor(
        dictionary_path=dict_path,
        fuzzy_threshold=75,
        context_weight=0.3
    )

    if model_path.exists():
        processor.load_model(model_path)
    else:
        print("Context model not found. Building...")
        build_full_model()
        processor.load_model(model_path)

    # Read input
    text = Path(input_path).read_text(encoding='utf-8')

    if verbose:
        print(f"\nProcessing: {input_path}")
        print(f"Words: {len(text.split())}")
        print("\nCorrections:")

    corrected, corrections = processor.process_text(text, verbose=verbose)

    if verbose:
        print(f"\nTotal corrections: {len(corrections)}")

    # Save output
    if output_path:
        Path(output_path).write_text(corrected, encoding='utf-8')
        print(f"\nSaved to: {output_path}")

    return corrected


if __name__ == "__main__":
    import argparse
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Context-aware OCR post-processor with fuzzy matching"
    )
    parser.add_argument("--build", action="store_true",
                        help="Build dictionary and context model")
    parser.add_argument("--input", "-i", type=str,
                        help="Input OCR text file")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for corrected text")
    parser.add_argument("--text", "-t", type=str,
                        help="Direct text input")
    parser.add_argument("--threshold", type=int, default=75,
                        help="Fuzzy matching threshold (0-100)")
    parser.add_argument("--context-weight", type=float, default=0.3,
                        help="Weight for context scoring (0-1)")

    args = parser.parse_args()

    if args.build:
        print("Building full model...")
        build_full_model()
        sys.exit(0)

    if args.input:
        corrected = process_ocr_file(
            args.input,
            args.output,
            verbose=True
        )

        if not args.output:
            print("\n" + "=" * 50)
            print("CORRECTED TEXT:")
            print("=" * 50)
            print(corrected)

    elif args.text:
        base_dir = Path(__file__).parent
        dict_path = base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt"
        model_path = base_dir / "ocr_context_model.pkl"

        processor = ContextAwarePostProcessor(
            dictionary_path=dict_path,
            fuzzy_threshold=args.threshold,
            context_weight=args.context_weight
        )

        if model_path.exists():
            processor.load_model(model_path)

        corrected, corrections = processor.process_text(args.text, verbose=True)
        print(f"\nOriginal: {args.text}")
        print(f"Corrected: {corrected}")

    else:
        print("Usage:")
        print("  1. Download dictionary:  python download_persian_dict.py")
        print("  2. Build context model:  python post_process_context.py --build")
        print("  3. Process file:         python post_process_context.py -i ocr_output.txt -o corrected.txt")
        print("  4. Process text:         python post_process_context.py -t 'your text here'")
