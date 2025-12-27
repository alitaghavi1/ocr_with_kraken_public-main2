"""
Post-process Kraken OCR output using the trained character classifier.

This script can help validate or correct OCR output by checking
individual characters against your trained classifier.
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

# Paths
MODEL_PATH = Path("C:/AR/python_utils/training_output/models/best_model.pth")
LABEL_MAPPING_PATH = Path("C:/AR/python_utils/training_output/models/label_mapping.json")

class CharacterClassifier:
    """Wrapper for the trained character classifier"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.idx_to_label = {}
        self.label_to_idx = {}

    def load(self):
        """Load the model and label mapping"""
        print(f"Loading model from {MODEL_PATH}...")

        # Load label mapping
        with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping = json.load(f)

        self.label_to_idx = mapping.get('label_to_idx', mapping)
        self.idx_to_label = {int(v): k for k, v in self.label_to_idx.items()}

        print(f"Loaded {len(self.idx_to_label)} character classes")

        # Load model
        # Note: You'll need to define the same architecture as your training
        # This is a placeholder - adjust based on your actual model architecture
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            print(f"Model loaded successfully")
            print(f"Model type: {type(checkpoint)}")

            # If it's a state dict, you need to create the model first
            if isinstance(checkpoint, dict):
                print("Model keys:", list(checkpoint.keys())[:10])
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        return True

    def predict(self, image):
        """Predict the character in an image"""
        # Placeholder - implement based on your model architecture
        pass

def analyze_dataset():
    """Analyze the character dataset"""
    print("Analyzing character dataset...")

    with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    label_to_idx = mapping.get('label_to_idx', mapping)

    # Categorize labels
    single_chars = []
    words = []
    numbers = []

    for label in label_to_idx.keys():
        if label.isdigit():
            numbers.append(label)
        elif len(label) == 1:
            single_chars.append(label)
        else:
            words.append(label)

    print(f"\nDataset Analysis:")
    print(f"  Total labels: {len(label_to_idx)}")
    print(f"  Single characters: {len(single_chars)}")
    print(f"  Words/phrases: {len(words)}")
    print(f"  Numbers: {len(numbers)}")

    print(f"\nSample single characters: {single_chars[:20]}")
    print(f"\nSample words: {words[:20]}")

    return {
        'single_chars': single_chars,
        'words': words,
        'numbers': numbers
    }

def main():
    print("=" * 50)
    print("Character Classifier Analysis")
    print("=" * 50)

    # Analyze the dataset
    analysis = analyze_dataset()

    # Try to load the model
    classifier = CharacterClassifier()
    classifier.load()

if __name__ == "__main__":
    main()
