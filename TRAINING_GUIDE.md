# Kraken OCR Fine-Tuning Guide for Handwritten Data

This guide explains how to fine-tune a Kraken OCR model with your own handwritten data.

## Quick Start

```bash
# 1. Prepare your training data
#    Put line images + transcriptions in handwritten_training_data/
#    Each image needs a .gt.txt file with the correct text

# 2. Train the model
run_training.bat

# 3. Test the trained model
.venv\Scripts\python.exe ocr_image.py your_image.jpg
```

## Training Approaches

### Option 1: Train from Scratch (Recommended for Handwritten Data)

Best when your handwriting is unique and significantly different from printed text.

```bash
# Requires 1000+ training samples
train_from_scratch.bat

# Or via Python:
.venv\Scripts\python.exe train.py --from-scratch
```

### Option 2: Fine-tune from Base Model

Start with a pre-trained model and adapt it to your handwriting. Good when you have limited data (200-500 samples).

```bash
# 1. Download a base model
.venv\Scripts\python.exe download_base_model.py --list
.venv\Scripts\python.exe download_base_model.py arabic_best

# 2. Set BASE_MODEL in train.py to the downloaded model path
# 3. Run training
finetune_from_base.bat
```

### Option 3: Continue from Checkpoint

Resume training from a previous session.

```bash
run_training.bat --continue
```

## Preparing Training Data

### Data Format

Kraken expects pairs of image + text files:
```
handwritten_training_data/
    line001.png          # Image of one line of handwritten text
    line001.gt.txt       # Contains: "exact transcription of line001.png"
    line002.png
    line002.gt.txt
    ...
```

### From Page Images

If you have full page scans, segment them into lines first:

```bash
.venv\Scripts\python.exe prepare_handwritten_data.py segment pages_folder/ output_folder/
```

This creates line images with empty .gt.txt files. You then manually transcribe each line.

### From Pre-cut Line Images

If you already have individual line images:

```bash
.venv\Scripts\python.exe prepare_handwritten_data.py from-lines lines_folder/ handwritten_training_data/
```

### Validate Your Data

Check that all images have proper transcriptions:

```bash
.venv\Scripts\python.exe prepare_handwritten_data.py validate handwritten_training_data/
```

### View Statistics

See what characters are in your training data:

```bash
.venv\Scripts\python.exe prepare_handwritten_data.py stats handwritten_training_data/
```

## Training Parameters

Edit `train.py` to customize training:

| Parameter | Default | Description |
|-----------|---------|-------------|
| BATCH_SIZE | 8 | Reduce to 4 or 2 if GPU memory issues |
| EPOCHS | 50 | Maximum training epochs |
| LEARNING_RATE | 0.0001 | Lower = more stable, slower learning |
| EARLY_STOPPING | 10 | Stop if no improvement for N epochs |
| DEVICE | cuda:0 | Use "cpu" if no GPU |
| USE_AUGMENTATION | True | Recommended for handwritten data |

## Monitoring Training

Training progress is logged to `training_log.txt`:

```bash
# On Windows, open the file in a text editor and refresh
# Or use PowerShell:
Get-Content training_log.txt -Wait
```

Key metrics to watch:
- **val_accuracy**: Should increase over time
- **val_CER**: Character Error Rate - should decrease
- **val_WER**: Word Error Rate - should decrease

## Tips for Better Results

### Data Quality
- Use high-resolution images (300+ DPI)
- Ensure consistent lighting and contrast
- Include diverse samples (different words, characters)
- Avoid blurry or cut-off text

### Data Quantity
- **Fine-tuning**: 200-500 samples minimum
- **From scratch**: 1000+ samples recommended
- More data = better results

### For Arabic/Persian Handwriting
- Set text direction to `horizontal-rl` (right-to-left)
- Include all character forms (initial, medial, final, isolated)
- Include common ligatures

### Training
- Start with a lower learning rate (0.0001) for fine-tuning
- Use data augmentation (enabled by default)
- Monitor validation metrics - stop if overfitting

## Troubleshooting

### "CUDA out of memory"
Reduce BATCH_SIZE in train.py (try 4, then 2)

### Training doesn't improve
- Add more diverse training data
- Check that transcriptions are correct
- Try lower learning rate

### Model predicts wrong characters
- Ensure all characters in your handwriting are represented in training data
- Add more samples of problematic characters

### Encoding errors on Windows
Use `run_training.py` which handles UTF-8 properly and logs to file

## Files Reference

| File | Description |
|------|-------------|
| `train.py` | Main training configuration |
| `run_training.py` | Training runner with logging |
| `run_training.bat` | Windows batch file for training |
| `train_from_scratch.bat` | Train new model |
| `finetune_from_base.bat` | Fine-tune from base model |
| `continue_training.bat` | Continue from checkpoint |
| `download_base_model.py` | Download Kraken models |
| `prepare_handwritten_data.py` | Data preparation tools |
| `training_log.txt` | Training output log |
| `models/fine_tuned_best.mlmodel` | Best trained model |

## Using the Trained Model

After training, use your model for OCR:

```bash
# Single image
.venv\Scripts\python.exe ocr_image.py image.jpg --print

# Or use the OCR pipeline
python ocr_pipeline.py -m models/fine_tuned_best.mlmodel
```
