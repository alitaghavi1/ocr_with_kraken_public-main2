"""
Enhanced OCR Post-Processor with:
1. Character Confusion Matrix - knows common Persian/Arabic OCR errors
2. Kraken Confidence Integration - uses per-character confidence scores
3. Multi-Hypothesis Processing - considers alternative readings

Usage:
    # With Kraken output (best)
    processor = EnhancedPostProcessor(dictionary_path="dictionaries/persian_dictionary_ganjoor.txt")
    corrected = processor.process_with_confidence(text, confidences)

    # With raw text only
    corrected = processor.process_text(text)
"""

import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Any
import pickle

try:
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import Levenshtein
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        FUZZY_LIB = None
        print("WARNING: No fuzzy matching library. Install: pip install rapidfuzz")


# =============================================================================
# PERSIAN/ARABIC CHARACTER CONFUSION MATRIX
# =============================================================================
# These are characters commonly confused by OCR due to visual similarity

CONFUSION_MATRIX = {
    # Dots confusion (most common)
    'ب': ['پ', 'ت', 'ث', 'ن', 'ی'],      # ba
    'پ': ['ب', 'ت', 'ث', 'چ'],            # pe
    'ت': ['ب', 'پ', 'ث', 'ن'],            # te
    'ث': ['ب', 'پ', 'ت'],                 # se
    'ج': ['چ', 'ح', 'خ'],                 # jim
    'چ': ['ج', 'ح', 'خ'],                 # che
    'ح': ['ج', 'چ', 'خ'],                 # he
    'خ': ['ج', 'چ', 'ح'],                 # khe
    'د': ['ذ'],                           # dal
    'ذ': ['د'],                           # zal
    'ر': ['ز', 'ژ', 'و'],                 # re
    'ز': ['ر', 'ژ'],                      # ze
    'ژ': ['ز', 'ر'],                      # zhe
    'س': ['ش'],                           # sin
    'ش': ['س'],                           # shin
    'ص': ['ض'],                           # sad
    'ض': ['ص'],                           # zad
    'ط': ['ظ'],                           # ta
    'ظ': ['ط'],                           # za
    'ع': ['غ'],                           # eyn
    'غ': ['ع'],                           # gheyn
    'ف': ['ق'],                           # fe
    'ق': ['ف'],                           # ghaf
    'ک': ['گ', 'ك'],                      # kaf
    'گ': ['ک', 'ك'],                      # gaf
    'ل': ['ا', 'لا'],                     # lam
    'م': ['ن'],                           # mim
    'ن': ['ب', 'ت', 'ث', 'م', 'ی'],       # nun
    'و': ['ر', 'ؤ'],                      # vav
    'ه': ['ة', 'ۀ', 'ھ'],                 # he
    'ی': ['ي', 'ى', 'ئ', 'ب', 'ت', 'ن'],  # ye

    # Arabic-specific
    'ك': ['ک', 'گ'],                      # Arabic kaf
    'ي': ['ی', 'ى'],                      # Arabic ye
    'ة': ['ه', 'ۀ'],                      # ta marbuta
    'أ': ['ا', 'إ', 'آ'],                 # alef hamza
    'إ': ['ا', 'أ', 'آ'],                 # alef hamza below
    'آ': ['ا', 'أ', 'إ'],                 # alef madda
    'ؤ': ['و', 'ء'],                      # vav hamza
    'ئ': ['ی', 'ء'],                      # ye hamza

    # Digits (Persian vs Arabic)
    '۴': ['۵', '4'],                      # 4
    '۵': ['۴', '5'],                      # 5
    '۶': ['6'],                           # 6
    '٤': ['٥', '۴', '۵'],                 # Arabic 4
    '٥': ['٤', '۴', '۵'],                 # Arabic 5
    '٦': ['۶', '6'],                      # Arabic 6
}

# Build reverse mapping: for each char, what chars might OCR produce instead?
OCR_MIGHT_PRODUCE = defaultdict(set)
for correct_char, confused_chars in CONFUSION_MATRIX.items():
    for confused in confused_chars:
        OCR_MIGHT_PRODUCE[confused].add(correct_char)
    # Also add self
    OCR_MIGHT_PRODUCE[correct_char].add(correct_char)


def confusion_distance(word1: str, word2: str) -> float:
    """
    Calculate edit distance weighted by confusion likelihood.
    Substitutions between confused characters cost less than random substitutions.

    Returns a score where lower = more similar (like Levenshtein).
    """
    if word1 == word2:
        return 0.0

    len1, len2 = len(word1), len(word2)

    # Simple case: length difference
    if abs(len1 - len2) > 3:
        return float(max(len1, len2))

    # Dynamic programming with weighted costs
    # Cost matrix: dp[i][j] = cost to transform word1[:i] to word2[:j]
    dp = [[0.0] * (len2 + 1) for _ in range(len1 + 1)]

    # Base cases
    for i in range(len1 + 1):
        dp[i][0] = i * 1.0  # Deletions cost 1.0
    for j in range(len2 + 1):
        dp[0][j] = j * 1.0  # Insertions cost 1.0

    # Fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            c1, c2 = word1[i-1], word2[j-1]

            if c1 == c2:
                # Same character, no cost
                substitution_cost = 0.0
            elif c2 in CONFUSION_MATRIX.get(c1, []) or c1 in CONFUSION_MATRIX.get(c2, []):
                # Known confusion pair - low cost
                substitution_cost = 0.3
            else:
                # Random substitution - full cost
                substitution_cost = 1.0

            dp[i][j] = min(
                dp[i-1][j] + 1.0,              # Deletion
                dp[i][j-1] + 1.0,              # Insertion
                dp[i-1][j-1] + substitution_cost  # Substitution
            )

    return dp[len1][len2]


def confusion_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity score (0-100) accounting for character confusions.
    Higher = more similar.
    """
    if word1 == word2:
        return 100.0

    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 100.0

    distance = confusion_distance(word1, word2)
    similarity = max(0, 100 * (1 - distance / max_len))

    return similarity


class EnhancedPostProcessor:
    """
    Enhanced OCR post-processor with:
    - Character confusion awareness
    - Confidence score integration
    - Multi-hypothesis support
    """

    def __init__(self,
                 dictionary_path: Optional[str] = None,
                 context_model_path: Optional[str] = None,
                 min_word_length: int = 2,
                 fuzzy_threshold: float = 70.0,
                 confusion_threshold: float = 65.0,
                 confidence_threshold: float = 0.8,
                 context_weight: float = 0.2):
        """
        Initialize enhanced post-processor.

        Args:
            dictionary_path: Path to word dictionary file
            context_model_path: Path to bigram context model (.pkl)
            min_word_length: Minimum word length to attempt correction
            fuzzy_threshold: Standard fuzzy matching threshold (0-100)
            confusion_threshold: Threshold for confusion-aware matching (0-100)
            confidence_threshold: Chars below this confidence get correction priority
            context_weight: Weight for context scoring (0-1)
        """
        self.min_word_length = min_word_length
        self.fuzzy_threshold = fuzzy_threshold
        self.confusion_threshold = confusion_threshold
        self.confidence_threshold = confidence_threshold
        self.context_weight = context_weight

        self.dictionary = set()
        self.word_freq = Counter()
        self.bigrams = defaultdict(Counter)
        self.total_bigrams = 0

        if dictionary_path:
            self.load_dictionary(dictionary_path)

        if context_model_path:
            self.load_context_model(context_model_path)

    def load_dictionary(self, path: str):
        """Load word dictionary."""
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

    def load_context_model(self, path: str):
        """Load bigram context model."""
        path = Path(path)
        if not path.exists():
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.bigrams = defaultdict(Counter, {k: Counter(v) for k, v in data.get('bigrams', {}).items()})
        self.word_freq = Counter(data.get('word_freq', {}))
        self.total_bigrams = data.get('total_bigrams', 0)

        print(f"Loaded context model: {self.total_bigrams:,} bigrams")

    def build_dictionary_from_ganjoor(self, ganjoor_dir: str):
        """Build dictionary from Ganjoor text files."""
        ganjoor_dir = Path(ganjoor_dir)
        if not ganjoor_dir.exists():
            print(f"Ganjoor directory not found: {ganjoor_dir}")
            return

        print("Building dictionary from Ganjoor texts...")

        for txt_file in ganjoor_dir.glob("*.txt"):
            try:
                text = txt_file.read_text(encoding='utf-8', errors='ignore')
                words = re.findall(r'[\u0600-\u06FF]+', text)
                for word in words:
                    if len(word) >= self.min_word_length:
                        self.dictionary.add(word)
                        self.word_freq[word] += 1
            except:
                pass

        print(f"Built dictionary with {len(self.dictionary):,} unique words")

    def _normalize(self, word: str) -> Optional[str]:
        """Normalize word for matching."""
        word = re.sub(r'^[^\u0600-\u06FF]+|[^\u0600-\u06FF]+$', '', word)
        return word if len(word) >= self.min_word_length else None

    def get_context_score(self, prev_word: str, word: str, next_word: str) -> float:
        """Get bigram-based context score (0-100)."""
        if self.total_bigrams == 0:
            return 50.0

        score = 0.0
        count = 0

        word_norm = self._normalize(word) if word else None

        if prev_word:
            prev_norm = self._normalize(prev_word)
            if prev_norm and prev_norm in self.bigrams:
                following = self.bigrams[prev_norm]
                if word_norm and word_norm in following:
                    freq = following[word_norm] / max(1, sum(following.values()))
                    score += min(100, freq * 500)
                count += 1

        if next_word and word_norm:
            next_norm = self._normalize(next_word)
            if word_norm in self.bigrams:
                following = self.bigrams[word_norm]
                if next_norm and next_norm in following:
                    freq = following[next_norm] / max(1, sum(following.values()))
                    score += min(100, freq * 500)
                count += 1

        return score / max(1, count) if count > 0 else 50.0

    def generate_confusion_variants(self, word: str, max_variants: int = 20) -> List[str]:
        """
        Generate possible intended words by applying confusion substitutions.

        For each character that has known confusions, generate variants
        with that character replaced.
        """
        variants = set()
        variants.add(word)

        # Single character substitutions
        for i, char in enumerate(word):
            if char in CONFUSION_MATRIX:
                for replacement in CONFUSION_MATRIX[char]:
                    variant = word[:i] + replacement + word[i+1:]
                    variants.add(variant)

        # Also check what might have produced each character
        for i, char in enumerate(word):
            if char in OCR_MIGHT_PRODUCE:
                for original in OCR_MIGHT_PRODUCE[char]:
                    variant = word[:i] + original + word[i+1:]
                    variants.add(variant)

        # Limit variants
        return list(variants)[:max_variants]

    def find_candidates(self,
                        word: str,
                        char_confidences: Optional[List[float]] = None,
                        max_candidates: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Find correction candidates for a word.

        Uses:
        1. Dictionary lookup
        2. Confusion-aware variant generation
        3. Fuzzy matching with confusion weighting
        4. Confidence-guided scoring

        Returns:
            List of (candidate, score, debug_info)
        """
        if not self.dictionary:
            return [(word, 100.0, {'source': 'no_dict'})]

        # Already in dictionary?
        if word in self.dictionary:
            return [(word, 100.0, {'source': 'in_dict'})]

        candidates = []

        # 1. Generate confusion variants and check dictionary
        variants = self.generate_confusion_variants(word)
        for variant in variants:
            if variant in self.dictionary and variant != word:
                score = confusion_similarity(word, variant)
                if score >= self.confusion_threshold:
                    candidates.append((variant, score, {
                        'source': 'confusion_variant',
                        'original': word,
                        'confusion_sim': score
                    }))

        # 2. Fuzzy matching against dictionary
        if FUZZY_LIB:
            # Use confusion-aware similarity for ranking
            fuzzy_matches = process.extract(
                word,
                self.dictionary,
                scorer=fuzz.ratio,
                limit=max_candidates * 2
            )

            for match in fuzzy_matches:
                candidate, fuzzy_score = match[0], match[1]
                if candidate == word:
                    continue

                # Re-score with confusion awareness
                conf_score = confusion_similarity(word, candidate)

                # Combined score (favor confusion-aware matches)
                combined = 0.4 * fuzzy_score + 0.6 * conf_score

                if combined >= self.confusion_threshold:
                    # Check if already added from variants
                    if not any(c[0] == candidate for c in candidates):
                        candidates.append((candidate, combined, {
                            'source': 'fuzzy',
                            'fuzzy_score': fuzzy_score,
                            'confusion_score': conf_score,
                            'combined': combined
                        }))

        # 3. Boost candidates based on character confidence
        if char_confidences and len(char_confidences) == len(word):
            low_conf_positions = [i for i, c in enumerate(char_confidences)
                                  if c < self.confidence_threshold]

            for i, (candidate, score, info) in enumerate(candidates):
                if len(candidate) == len(word):
                    # Check if differences align with low-confidence positions
                    diff_positions = [j for j in range(len(word))
                                      if j < len(candidate) and word[j] != candidate[j]]

                    overlap = len(set(diff_positions) & set(low_conf_positions))
                    if overlap > 0:
                        # Boost score for matches that fix low-confidence chars
                        boost = 5 * overlap
                        candidates[i] = (candidate, min(100, score + boost), {
                            **info,
                            'confidence_boost': boost,
                            'low_conf_positions': low_conf_positions
                        })

        # Sort by score
        candidates.sort(key=lambda x: -x[1])

        # Add word frequency bonus
        final_candidates = []
        for candidate, score, info in candidates[:max_candidates]:
            freq_bonus = 0
            if self.word_freq.get(candidate, 0) > 10:
                freq_bonus = 3
            elif self.word_freq.get(candidate, 0) > 100:
                freq_bonus = 5

            final_score = min(100, score + freq_bonus)
            final_candidates.append((candidate, final_score, {**info, 'freq_bonus': freq_bonus}))

        if not final_candidates:
            return [(word, 0.0, {'source': 'no_match'})]

        return final_candidates

    def correct_word(self,
                     word: str,
                     char_confidences: Optional[List[float]] = None,
                     prev_word: Optional[str] = None,
                     next_word: Optional[str] = None) -> Tuple[str, bool, Optional[Dict]]:
        """
        Correct a single word.

        Args:
            word: The word to correct
            char_confidences: Per-character confidence scores from Kraken
            prev_word: Previous word for context
            next_word: Next word for context

        Returns:
            (corrected_word, was_corrected, debug_info)
        """
        if len(word) < self.min_word_length:
            return word, False, None

        # Already in dictionary?
        if word in self.dictionary:
            return word, False, None

        # Get candidates
        candidates = self.find_candidates(word, char_confidences)

        if not candidates or candidates[0][1] == 0:
            return word, False, None

        # Score with context
        best_candidate = word
        best_score = 0.0
        best_info = None

        for candidate, base_score, info in candidates:
            # Context score
            context_score = self.get_context_score(prev_word, candidate, next_word)

            # Combined score
            final_score = (1 - self.context_weight) * base_score + self.context_weight * context_score

            if final_score > best_score:
                best_score = final_score
                best_candidate = candidate
                best_info = {
                    **info,
                    'base_score': base_score,
                    'context_score': context_score,
                    'final_score': final_score,
                    'prev_word': prev_word,
                    'next_word': next_word
                }

        # Apply threshold
        threshold = self.confusion_threshold

        # Lower threshold if we have confidence info showing low confidence
        if char_confidences:
            avg_conf = sum(char_confidences) / len(char_confidences)
            if avg_conf < self.confidence_threshold:
                threshold = max(50, threshold - 10)

        was_corrected = best_candidate != word and best_score >= threshold

        return best_candidate, was_corrected, best_info if was_corrected else None

    def process_text(self, text: str, verbose: bool = False) -> Tuple[str, List[Dict]]:
        """
        Process text without confidence information.

        Returns:
            (corrected_text, list_of_corrections)
        """
        words = text.split()
        corrected_words = []
        corrections = []

        for i, word in enumerate(words):
            prev_word = words[i-1] if i > 0 else None
            next_word = words[i+1] if i < len(words) - 1 else None

            corrected, was_corrected, info = self.correct_word(
                word, None, prev_word, next_word
            )

            corrected_words.append(corrected)

            if was_corrected and info:
                correction = {'original': word, 'corrected': corrected, **info}
                corrections.append(correction)
                if verbose:
                    self._print_correction(correction)

        return ' '.join(corrected_words), corrections

    def process_with_confidence(self,
                                 text: str,
                                 char_confidences: List[List[float]],
                                 verbose: bool = False) -> Tuple[str, List[Dict]]:
        """
        Process text with per-character confidence from Kraken.

        Args:
            text: OCR output text
            char_confidences: List of confidence lists, one per word
            verbose: Print corrections

        Returns:
            (corrected_text, list_of_corrections)
        """
        words = text.split()

        # Align confidences with words
        if len(char_confidences) != len(words):
            print(f"Warning: confidence count ({len(char_confidences)}) != word count ({len(words)})")
            return self.process_text(text, verbose)

        corrected_words = []
        corrections = []

        for i, (word, word_conf) in enumerate(zip(words, char_confidences)):
            prev_word = words[i-1] if i > 0 else None
            next_word = words[i+1] if i < len(words) - 1 else None

            corrected, was_corrected, info = self.correct_word(
                word, word_conf, prev_word, next_word
            )

            corrected_words.append(corrected)

            if was_corrected and info:
                correction = {
                    'original': word,
                    'corrected': corrected,
                    'word_confidence': sum(word_conf)/len(word_conf) if word_conf else None,
                    **info
                }
                corrections.append(correction)
                if verbose:
                    self._print_correction(correction)

        return ' '.join(corrected_words), corrections

    def process_kraken_records(self,
                                records: List[Any],
                                verbose: bool = False) -> Tuple[str, List[Dict]]:
        """
        Process Kraken OCR records directly.

        Args:
            records: List of kraken ocr_record objects
            verbose: Print corrections

        Returns:
            (corrected_text, list_of_corrections)
        """
        all_words = []
        all_confidences = []

        for record in records:
            # Split line into words and align confidences
            line_text = record.prediction
            line_conf = record.confidences

            words = line_text.split()
            if not words:
                continue

            # Distribute confidences across words
            # This is approximate since we don't have word boundaries
            char_idx = 0
            for word in words:
                word_len = len(word)
                if char_idx + word_len <= len(line_conf):
                    word_conf = line_conf[char_idx:char_idx + word_len]
                else:
                    word_conf = [0.5] * word_len  # Default if mismatch

                all_words.append(word)
                all_confidences.append(word_conf)
                char_idx += word_len + 1  # +1 for space

        if not all_words:
            return "", []

        # Now process with confidence
        text = ' '.join(all_words)
        return self.process_with_confidence(text, all_confidences, verbose)

    def _print_correction(self, correction: Dict):
        """Print a correction for verbose output."""
        try:
            orig = correction.get('original', '?')
            corr = correction.get('corrected', '?')
            score = correction.get('final_score', 0)
            source = correction.get('source', 'unknown')
            conf = correction.get('word_confidence')

            conf_str = f", conf:{conf:.2f}" if conf else ""
            print(f"  '{orig}' -> '{corr}' (score:{score:.1f}, source:{source}{conf_str})")
        except:
            print(f"  [correction made]")

    def save_dictionary(self, path: str):
        """Save dictionary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sorted_words = sorted(self.word_freq.items(), key=lambda x: -x[1])

        with open(path, 'w', encoding='utf-8') as f:
            for word, freq in sorted_words:
                f.write(f"{word}\n")

        print(f"Saved {len(sorted_words)} words to {path}")


# =============================================================================
# MULTI-HYPOTHESIS PROCESSOR
# =============================================================================

class MultiHypothesisProcessor:
    """
    Process multiple OCR hypotheses to find best reading.

    When Kraken outputs multiple possible readings (via beam search),
    this class evaluates them using dictionary, confusion matrix, and context.
    """

    def __init__(self, post_processor: EnhancedPostProcessor):
        self.pp = post_processor

    def score_hypothesis(self,
                         text: str,
                         base_score: float = 0.0) -> Tuple[float, Dict]:
        """
        Score a complete hypothesis.

        Args:
            text: The hypothesis text
            base_score: Initial score (e.g., from decoder confidence)

        Returns:
            (score, debug_info)
        """
        words = text.split()

        # Dictionary coverage
        in_dict = sum(1 for w in words if w in self.pp.dictionary)
        dict_coverage = in_dict / max(1, len(words))

        # Context coherence (bigram likelihood)
        context_score = 0.0
        if len(words) > 1:
            for i in range(len(words) - 1):
                context_score += self.pp.get_context_score(
                    words[i], words[i+1],
                    words[i+2] if i+2 < len(words) else None
                )
            context_score /= len(words) - 1
        else:
            context_score = 50.0

        # Combined score
        final_score = (
            0.3 * base_score +
            0.4 * (dict_coverage * 100) +
            0.3 * context_score
        )

        return final_score, {
            'base_score': base_score,
            'dict_coverage': dict_coverage,
            'context_score': context_score,
            'in_dict_words': in_dict,
            'total_words': len(words)
        }

    def select_best_hypothesis(self,
                                hypotheses: List[Tuple[str, float]],
                                verbose: bool = False) -> Tuple[str, Dict]:
        """
        Select best hypothesis from multiple candidates.

        Args:
            hypotheses: List of (text, decoder_score) tuples
            verbose: Print scoring details

        Returns:
            (best_text, debug_info)
        """
        if not hypotheses:
            return "", {}

        if len(hypotheses) == 1:
            return hypotheses[0][0], {'single_hypothesis': True}

        scored = []
        for text, decoder_score in hypotheses:
            score, info = self.score_hypothesis(text, decoder_score)
            scored.append((text, score, info))

            if verbose:
                print(f"  Hypothesis: '{text[:50]}...' -> score:{score:.1f}")

        # Sort by score
        scored.sort(key=lambda x: -x[1])

        best_text, best_score, best_info = scored[0]

        return best_text, {
            'best_score': best_score,
            'num_hypotheses': len(hypotheses),
            'runner_up_score': scored[1][1] if len(scored) > 1 else None,
            **best_info
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Enhanced OCR post-processor with confusion matrix and confidence integration"
    )
    parser.add_argument("--build-dict", action="store_true",
                        help="Build dictionary from Ganjoor texts")
    parser.add_argument("--input", "-i", type=str,
                        help="Input OCR text file")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for corrected text")
    parser.add_argument("--text", "-t", type=str,
                        help="Direct text input")
    parser.add_argument("--dict", "-d", type=str,
                        help="Path to dictionary file")
    parser.add_argument("--threshold", type=float, default=65.0,
                        help="Confusion matching threshold (0-100)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print corrections")

    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Build dictionary from Ganjoor
    if args.build_dict:
        ganjoor_dir = base_dir / "ganjoor_texts"
        if not ganjoor_dir.exists():
            print(f"Ganjoor texts not found at: {ganjoor_dir}")
            print("Run: python download_ganjoor.py")
            sys.exit(1)

        processor = EnhancedPostProcessor()
        processor.build_dictionary_from_ganjoor(ganjoor_dir)

        dict_path = base_dir / "dictionaries" / "enhanced_persian_dict.txt"
        processor.save_dictionary(dict_path)
        sys.exit(0)

    # Load processor
    dict_path = args.dict or base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt"
    context_path = base_dir / "ocr_context_model.pkl"

    processor = EnhancedPostProcessor(
        dictionary_path=str(dict_path) if Path(dict_path).exists() else None,
        context_model_path=str(context_path) if Path(context_path).exists() else None,
        confusion_threshold=args.threshold
    )

    # Get input text
    if args.input:
        text = Path(args.input).read_text(encoding='utf-8')
    elif args.text:
        text = args.text
    else:
        print("Usage:")
        print("  Build dictionary:  python post_process_enhanced.py --build-dict")
        print("  Process file:      python post_process_enhanced.py -i input.txt -o output.txt -v")
        print("  Process text:      python post_process_enhanced.py -t 'your text' -v")
        sys.exit(0)

    # Process
    print(f"\nProcessing ({len(text.split())} words)...")
    if args.verbose:
        print("Corrections:")

    corrected, corrections = processor.process_text(text, verbose=args.verbose)

    print(f"\nTotal corrections: {len(corrections)}")

    # Output
    if args.output:
        Path(args.output).write_text(corrected, encoding='utf-8')
        print(f"Saved to: {args.output}")
    else:
        print("\n" + "=" * 50)
        print("ORIGINAL:")
        print(text[:500] + "..." if len(text) > 500 else text)
        print("\nCORRECTED:")
        print(corrected[:500] + "..." if len(corrected) > 500 else corrected)


if __name__ == "__main__":
    main()
