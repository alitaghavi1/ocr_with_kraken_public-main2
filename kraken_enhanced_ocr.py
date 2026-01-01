"""
Enhanced Kraken OCR with Multi-Hypothesis Output and Post-Processing.

This module wraps Kraken to:
1. Extract multiple hypotheses (beam search alternatives)
2. Get per-character confidence scores
3. Apply enhanced post-processing

Usage:
    from kraken_enhanced_ocr import EnhancedOCR

    ocr = EnhancedOCR(model_path="models/my_model.mlmodel")
    result = ocr.recognize(image_path, apply_postprocess=True)

    print(result['text'])           # Best corrected text
    print(result['raw_text'])       # Raw OCR output
    print(result['hypotheses'])     # Alternative readings
    print(result['corrections'])    # What was corrected
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from PIL import Image

# Kraken imports
from kraken import blla, rpred
from kraken.lib import models
from kraken.containers import Segmentation

# Import enhanced post-processor
from post_process_enhanced import EnhancedPostProcessor, MultiHypothesisProcessor


@dataclass
class WordInfo:
    """Information about a recognized word."""
    text: str
    confidence: float
    char_confidences: List[float]
    position: Tuple[int, int, int, int]  # x1, y1, x2, y2
    alternatives: List[Tuple[str, float]]  # (text, score)


@dataclass
class LineInfo:
    """Information about a recognized line."""
    text: str
    words: List[WordInfo]
    avg_confidence: float
    raw_prediction: str
    raw_confidences: List[float]


@dataclass
class OCRResult:
    """Complete OCR result for an image."""
    text: str                    # Final corrected text
    raw_text: str                # Raw OCR output
    lines: List[LineInfo]        # Per-line information
    corrections: List[Dict]      # Corrections made
    hypotheses: List[Tuple[str, float]]  # Alternative readings


class EnhancedOCR:
    """
    Enhanced OCR with multi-hypothesis and post-processing.
    """

    def __init__(self,
                 model_path: str,
                 dictionary_path: Optional[str] = None,
                 context_model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize enhanced OCR.

        Args:
            model_path: Path to Kraken model file
            dictionary_path: Path to word dictionary (optional)
            context_model_path: Path to bigram context model (optional)
            device: Device to run on ("cpu" or "cuda:0")
        """
        self.device = device
        self.model = None
        self.post_processor = None
        self.multi_hyp_processor = None

        # Load Kraken model
        self._load_model(model_path)

        # Load post-processor
        self._load_post_processor(dictionary_path, context_model_path)

    def _load_model(self, model_path: str):
        """Load Kraken recognition model."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model: {model_path}")
        self.model = models.load_any(str(model_path), device=self.device)
        print(f"Model loaded on {self.device}")

    def _load_post_processor(self,
                              dictionary_path: Optional[str],
                              context_model_path: Optional[str]):
        """Load post-processor with dictionary and context model."""
        base_dir = Path(__file__).parent

        # Find dictionary
        if dictionary_path is None:
            possible_dicts = [
                base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt",
                base_dir / "dictionaries" / "enhanced_persian_dict.txt",
                base_dir / "ocr_dictionary.txt",
            ]
            for p in possible_dicts:
                if p.exists():
                    dictionary_path = str(p)
                    break

        # Find context model
        if context_model_path is None:
            ctx_path = base_dir / "ocr_context_model.pkl"
            if ctx_path.exists():
                context_model_path = str(ctx_path)

        # Create post-processor
        self.post_processor = EnhancedPostProcessor(
            dictionary_path=dictionary_path,
            context_model_path=context_model_path
        )

        # Create multi-hypothesis processor
        self.multi_hyp_processor = MultiHypothesisProcessor(self.post_processor)

    def segment(self, image: Image.Image) -> Segmentation:
        """Segment image into text lines."""
        # Use baseline segmentation
        return blla.segment(image)

    def recognize_line(self,
                        image: Image.Image,
                        bounds: Segmentation) -> List[LineInfo]:
        """
        Recognize text in segmented lines.

        Returns per-line information including confidences.
        """
        lines = []

        for record in rpred.rpred(self.model, image, bounds):
            # Extract line info
            line_text = record.prediction
            line_conf = record.confidences

            avg_conf = sum(line_conf) / len(line_conf) if line_conf else 0.0

            # Parse into words
            words = self._parse_words(line_text, line_conf, record.cuts)

            line_info = LineInfo(
                text=line_text,
                words=words,
                avg_confidence=avg_conf,
                raw_prediction=line_text,
                raw_confidences=list(line_conf)
            )
            lines.append(line_info)

        return lines

    def _parse_words(self,
                      text: str,
                      confidences: List[float],
                      cuts: List) -> List[WordInfo]:
        """Parse line into words with confidence info."""
        words = []
        word_texts = text.split()

        if not word_texts:
            return words

        # Distribute confidences across words
        char_idx = 0
        for word_text in word_texts:
            word_len = len(word_text)

            # Get confidence for this word's characters
            if char_idx + word_len <= len(confidences):
                word_conf = confidences[char_idx:char_idx + word_len]
            else:
                word_conf = [0.5] * word_len

            avg_conf = sum(word_conf) / len(word_conf) if word_conf else 0.0

            # Position (approximate from cuts if available)
            position = (0, 0, 0, 0)  # TODO: extract from cuts

            word_info = WordInfo(
                text=word_text,
                confidence=avg_conf,
                char_confidences=list(word_conf),
                position=position,
                alternatives=[]
            )
            words.append(word_info)

            char_idx += word_len + 1  # +1 for space

        return words

    def recognize(self,
                   image_path: str,
                   apply_postprocess: bool = True,
                   verbose: bool = False) -> OCRResult:
        """
        Full OCR pipeline with optional post-processing.

        Args:
            image_path: Path to image file
            apply_postprocess: Whether to apply enhanced post-processing
            verbose: Print progress and corrections

        Returns:
            OCRResult with text, confidence, and correction info
        """
        # Load image
        image = Image.open(image_path)
        if image.mode != 'L':
            image = image.convert('L')

        if verbose:
            print(f"Processing: {image_path}")
            print(f"Image size: {image.size}")

        # Segment
        if verbose:
            print("Segmenting...")
        bounds = self.segment(image)

        if verbose:
            print(f"Found {len(bounds.lines)} lines")

        # Recognize
        if verbose:
            print("Recognizing...")
        lines = self.recognize_line(image, bounds)

        # Combine lines into full text
        raw_text = '\n'.join(line.text for line in lines)

        # Collect all words with confidences for post-processing
        all_words = []
        all_word_confidences = []

        for line in lines:
            for word in line.words:
                all_words.append(word.text)
                all_word_confidences.append(word.char_confidences)

        # Apply post-processing
        corrections = []
        if apply_postprocess and self.post_processor:
            if verbose:
                print("Post-processing...")

            text_to_process = ' '.join(all_words)
            corrected_text, corrections = self.post_processor.process_with_confidence(
                text_to_process,
                all_word_confidences,
                verbose=verbose
            )

            # Restore line breaks
            final_text = self._restore_line_breaks(corrected_text, lines)
        else:
            final_text = raw_text

        # Build result
        result = OCRResult(
            text=final_text,
            raw_text=raw_text,
            lines=lines,
            corrections=corrections,
            hypotheses=[]  # TODO: implement beam search alternatives
        )

        if verbose:
            print(f"\nRecognition complete:")
            print(f"  Lines: {len(lines)}")
            print(f"  Words: {len(all_words)}")
            print(f"  Corrections: {len(corrections)}")

        return result

    def _restore_line_breaks(self, text: str, lines: List[LineInfo]) -> str:
        """Restore line breaks to match original structure."""
        words = text.split()
        result_lines = []

        word_idx = 0
        for line in lines:
            line_word_count = len(line.words)
            line_words = words[word_idx:word_idx + line_word_count]
            result_lines.append(' '.join(line_words))
            word_idx += line_word_count

        # Add any remaining words to last line
        if word_idx < len(words):
            if result_lines:
                result_lines[-1] += ' ' + ' '.join(words[word_idx:])
            else:
                result_lines.append(' '.join(words[word_idx:]))

        return '\n'.join(result_lines)

    def recognize_batch(self,
                         image_paths: List[str],
                         apply_postprocess: bool = True,
                         verbose: bool = False) -> List[OCRResult]:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths
            apply_postprocess: Whether to apply post-processing
            verbose: Print progress

        Returns:
            List of OCRResult objects
        """
        results = []

        for i, path in enumerate(image_paths):
            if verbose:
                print(f"\n[{i+1}/{len(image_paths)}] ", end="")

            try:
                result = self.recognize(path, apply_postprocess, verbose)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append(OCRResult(
                    text="",
                    raw_text="",
                    lines=[],
                    corrections=[],
                    hypotheses=[]
                ))

        return results


def get_word_alternatives(word: str,
                           char_confidences: List[float],
                           post_processor: EnhancedPostProcessor,
                           max_alternatives: int = 5) -> List[Tuple[str, float]]:
    """
    Generate alternative readings for a word based on confidence and confusion matrix.

    This simulates multi-hypothesis by generating likely alternatives
    for low-confidence characters using the confusion matrix.

    Args:
        word: The recognized word
        char_confidences: Per-character confidence scores
        post_processor: The enhanced post-processor
        max_alternatives: Maximum alternatives to generate

    Returns:
        List of (alternative_word, score) tuples
    """
    from post_process_enhanced import CONFUSION_MATRIX

    alternatives = [(word, 100.0)]

    # Find low-confidence positions
    low_conf_positions = [
        (i, conf) for i, conf in enumerate(char_confidences)
        if conf < post_processor.confidence_threshold
    ]

    # Sort by confidence (lowest first)
    low_conf_positions.sort(key=lambda x: x[1])

    # Generate alternatives by substituting confused characters
    for pos, conf in low_conf_positions[:3]:  # Top 3 low-confidence positions
        char = word[pos]

        if char in CONFUSION_MATRIX:
            for replacement in CONFUSION_MATRIX[char][:3]:  # Top 3 confusions
                alt_word = word[:pos] + replacement + word[pos+1:]

                # Score based on confidence boost for fixing low-conf char
                # and whether result is in dictionary
                score = (1 - conf) * 100  # Higher score for lower original confidence

                if alt_word in post_processor.dictionary:
                    score += 20  # Bonus for dictionary match

                alternatives.append((alt_word, score))

    # Sort by score and deduplicate
    seen = set()
    unique_alts = []
    for alt, score in sorted(alternatives, key=lambda x: -x[1]):
        if alt not in seen:
            seen.add(alt)
            unique_alts.append((alt, score))

    return unique_alts[:max_alternatives]


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description="Enhanced Kraken OCR with post-processing"
    )
    parser.add_argument("image", type=str, nargs="?",
                        help="Image file to process")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Path to Kraken model")
    parser.add_argument("--output", "-o", type=str,
                        help="Output file for text")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable post-processing")
    parser.add_argument("--dict", "-d", type=str,
                        help="Path to dictionary file")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda:0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress and corrections")
    parser.add_argument("--batch", "-b", type=str,
                        help="Process all images in directory")

    args = parser.parse_args()

    # Initialize OCR
    ocr = EnhancedOCR(
        model_path=args.model,
        dictionary_path=args.dict,
        device=args.device
    )

    # Process
    if args.batch:
        # Batch mode
        batch_dir = Path(args.batch)
        image_paths = list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.jpg"))
        print(f"Processing {len(image_paths)} images...")

        results = ocr.recognize_batch(
            [str(p) for p in image_paths],
            apply_postprocess=not args.no_postprocess,
            verbose=args.verbose
        )

        # Save results
        output_dir = Path(args.output) if args.output else batch_dir / "ocr_output"
        output_dir.mkdir(exist_ok=True)

        for path, result in zip(image_paths, results):
            out_file = output_dir / (path.stem + ".txt")
            out_file.write_text(result.text, encoding='utf-8')

        print(f"\nResults saved to: {output_dir}")

    elif args.image:
        # Single image mode
        result = ocr.recognize(
            args.image,
            apply_postprocess=not args.no_postprocess,
            verbose=args.verbose
        )

        print("\n" + "=" * 50)
        print("RESULT:")
        print("=" * 50)
        print(result.text)

        if args.output:
            Path(args.output).write_text(result.text, encoding='utf-8')
            print(f"\nSaved to: {args.output}")

        if result.corrections:
            print(f"\n{len(result.corrections)} corrections made")

    else:
        print("Usage:")
        print("  Single image:  python kraken_enhanced_ocr.py image.png -m model.mlmodel -v")
        print("  Batch:         python kraken_enhanced_ocr.py -b images_dir -m model.mlmodel -o output_dir")


if __name__ == "__main__":
    main()
