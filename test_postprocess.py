"""Simple test script for post-processing."""
import sys
from pathlib import Path

# Set encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from post_process_context import ContextAwarePostProcessor

base_dir = Path(__file__).parent
dict_path = base_dir / "dictionaries" / "persian_dictionary_ganjoor.txt"
model_path = base_dir / "ocr_context_model.pkl"

# Also add corpus vocabulary to dictionary
corpus_vocab = base_dir / "corpus_vocabulary.txt"

print("Loading processor...")
processor = ContextAwarePostProcessor(
    dictionary_path=dict_path,
    fuzzy_threshold=85,  # Higher threshold to avoid bad corrections
    context_weight=0.4   # More weight on context now that we have good data
)

# Add corpus vocabulary
if corpus_vocab.exists():
    print("Adding corpus vocabulary...")
    with open(corpus_vocab, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and len(word) >= 2:
                processor.dictionary.add(word)
    print(f"Dictionary size: {len(processor.dictionary):,}")

if model_path.exists():
    processor.load_model(model_path)

# Read input
input_path = base_dir / "OUTPUT" / "yamini.txt"
print(f"Reading: {input_path}")
text = input_path.read_text(encoding='utf-8')
print(f"Words: {len(text.split())}")

# Process without verbose to avoid encoding issues
print("Processing...")
corrected, corrections = processor.process_text(text, verbose=False)

print(f"Corrections made: {len(corrections)}")

# Save output
output_path = base_dir / "OUTPUT" / "yamini_corrected.txt"
output_path.write_text(corrected, encoding='utf-8')
print(f"Saved to: {output_path}")

# Save corrections log
log_path = base_dir / "OUTPUT" / "yamini_corrections.txt"
with open(log_path, 'w', encoding='utf-8') as f:
    for c in corrections:
        f.write(f"{c['original']} -> {c['candidate']} (fuzzy:{c['fuzzy']:.0f}, context:{c['context']:.0f})\n")
print(f"Corrections log: {log_path}")

print("Done!")
