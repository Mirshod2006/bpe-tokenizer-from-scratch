# Setup & Run Guide

Complete guide for setting up and training the BPE tokenizer.

## Quick Start

```bash
# Full pipeline (download data + train tokenizer)
python setup&run.py

# Or run steps separately:
python setup&run.py --setup    # Download TinyStories dataset
python setup&run.py --train    # Train tokenizer
```

## Command-Line Options

### Basic Commands

| Command | Description |
|---------|-------------|
| `python setup&run.py` | Run full pipeline (setup + train) |
| `python setup&run.py --setup` | Only setup (download data) |
| `python setup&run.py --train` | Only train (assumes data exists) |

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--vocab-size N` | Target vocabulary size | 50,000 |
| `--max-size MB` | Limit corpus size (MB) for faster training | No limit |
| `--no-preprocess` | Skip preprocessing, use existing processed data | False |

### Examples

```bash
# Train with smaller vocabulary
python setup&run.py --vocab-size 30000

# Fast training on first 100MB only
python setup&run.py --max-size 100

# Use existing preprocessed data
python setup&run.py --train --no-preprocess

# Download data only (for later training)
python setup&run.py --setup
```

## What It Does

### Setup Phase (`--setup`)

1. **Creates Directory Structure**
   - `data/raw/` - Raw downloaded datasets
   - `data/processed/` - Preprocessed corpus
   - `src/models/` - Trained model files

2. **Downloads TinyStories Dataset**
   - Training set (~500MB): `TinyStoriesV2-GPT4-train.txt`
   - Validation set (~50MB): `TinyStoriesV2-GPT4-valid.txt`
   - Source: [HuggingFace TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

### Training Phase (`--train`)

1. **Preprocessing Corpus**
   - Unicode normalization (NFC)
   - HTML entity decoding
   - URL/email normalization
   - Control character removal
   - Whitespace standardization
   - Saves to `data/processed/`

2. **Training BPE Tokenizer**
   - Initializes vocabulary with characters
   - Performs byte-pair encoding merges
   - Builds merge rules
   - Target vocab size: 50,000 (configurable)

3. **Saving Model**
   - `src/models/vocab.json` - Vocabulary mapping
   - `src/models/merges.json` - BPE merge rules
   - `src/models/tokenizer_config.json` - Configuration

4. **Testing**
   - Runs test encoding/decoding
   - Verifies tokenizer works correctly

## Output Files

After successful run:

```
src/models/
├── vocab.json              # Token ID to string mapping
├── merges.json             # BPE merge rules
└── tokenizer_config.json   # Tokenizer configuration

data/raw/
├── TinyStoriesV2-GPT4-train.txt  # Downloaded training data
└── TinyStoriesV2-GPT4-valid.txt  # Downloaded validation data

data/processed/
├── TinyStoriesV2-GPT4-train.txt  # Preprocessed training data
└── TinyStoriesV2-GPT4-valid.txt  # Preprocessed validation data (optional)
```

## Using the Trained Tokenizer

```python
from src.bpe.tokenizer import BPETokenizer

# Load trained tokenizer
tokenizer = BPETokenizer()
tokenizer.load(
    vocab_path="src/models/vocab.json",
    merges_path="src/models/merges.json"
)

# Encode text
text = "Hello, world! This is a test."
token_ids = tokenizer.encode(text)
print(f"Tokens: {token_ids}")

# Decode back
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")
assert decoded == text  # Should match!

# Get vocab size
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
```

## Performance Tips

### Fast Training (for testing)

```bash
# Train on first 100MB only (much faster)
python setup&run.py --max-size 100 --vocab-size 10000
```

### Full Production Training

```bash
# Use full dataset with large vocabulary
python setup&run.py --vocab-size 50000
```

### Skip Preprocessing (if already done)

```bash
# Preprocess once
python setup&run.py --train

# Later, retrain with different vocab size (reuse preprocessed data)
python setup&run.py --train --no-preprocess --vocab-size 30000
```

## Troubleshooting

### "Training data not found"

Run setup first:
```bash
python setup&run.py --setup
```

### "Preprocessed data not found"

Either:
1. Run with preprocessing: `python setup&run.py --train`
2. Remove `--no-preprocess` flag

### Out of Memory

Limit corpus size:
```bash
python setup&run.py --max-size 100  # Use first 100MB only
```

### Download fails

- Check internet connection
- Try downloading manually from URLs in `src/bpe/constants.py`
- Place files in `data/raw/` directory

### Module not found errors

Install dependencies:
```bash
pip install -r requirements.txt
```

## System Requirements

- **Python**: 3.7+
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk Space**: ~1GB for dataset + models
- **Internet**: Required for downloading dataset

## Dependencies

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `regex` - Unicode-aware pattern matching
- `requests` - Dataset downloading

## Architecture

### File Structure

```
setup&run.py                 # Main orchestration script
├── setup()                  # Download data, create directories
└── run_training_pipeline()  # Preprocess, train, save

scripts/
├── preprocess_corpus.py     # Text preprocessing filters
└── download_data.py         # Dataset download utilities

src/bpe/
├── tokenizer.py             # Main BPE tokenizer class
├── trainer.py               # BPE training logic
├── encoder.py               # Text to token IDs
├── decoder.py               # Token IDs to text
├── bpe_merger.py            # BPE merge operations
├── utils.py                 # Helper functions
└── constants.py             # Configuration constants
```

### Design Decision: One File

**Why keep setup and run in one file?**

✅ **Advantages:**
- Sequential operations (setup → train)
- Shared configuration
- Single entry point
- Clear pipeline flow
- Easy to understand

**Alternative: Split files**

If you prefer, you can split into:
- `setup.py` - Setup and data download
- `train.py` - Training pipeline
- `config.py` - Shared configuration

## Next Steps

After training:

1. **Test tokenizer** - See examples above
2. **Benchmark performance** - Run `scripts/benchmark.py`
3. **Try on custom data** - Modify corpus path
4. **Tune vocabulary size** - Experiment with `--vocab-size`
5. **Train language model** - Use tokenizer for downstream tasks

## References

- [TinyStories Paper](https://arxiv.org/abs/2305.07759)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
- [Preprocessing Filters Documentation](docs/preprocessing_filters.md)
