# Setup & Run Guide

Complete guide for setting up and training the BPE tokenizer.

## Quick Start

```bash
# Full pipeline (download data + train tokenizer)
python src/setup_and_run.py

# Or run steps separately:
python src/setup_and_run.py --setup    # Download TinyStories dataset
python src/setup_and_run.py --train    # Train tokenizer
```

## Command-Line Options

### Basic Commands

| Command | Description |
|---------|-------------|
| `python src/setup_and_run.py` | Run full pipeline (setup + train) |
| `python src/setup_and_run.py --setup` | Only setup (download data) |
| `python src/setup_and_run.py --train` | Only train (assumes data exists) |

### Advanced Options

| Option | Description | Default |
|--------|-------------|---------|
| `--vocab-size N` | Target vocabulary size | 50,000 |
| `--max-size MB` | Limit corpus size (MB) for faster training | No limit |
| `--no-preprocess` | Skip preprocessing, use existing processed data | False |

### Examples

```bash
# Train with smaller vocabulary
python src/setup_and_run.py --vocab-size 30000

# Fast training on first 100MB only
python src/setup_and_run.py --max-size 100

# Use existing preprocessed data
python src/setup_and_run.py --train --no-preprocess

# Download data only (for later training)
python src/setup_and_run.py --setup
```

## What It Does

### Setup Phase (`--setup`)

1. **Creates Directory Structure**
   - `data/raw/` — Raw downloaded datasets
   - `data/processed/` — Preprocessed corpus
   - `models/` — Trained model files

2. **Downloads TinyStories Dataset**
   - Training set (~500MB): `TinyStoriesV2-GPT4-train.txt`
   - Validation set (~50MB): `TinyStoriesV2-GPT4-valid.txt`
   - Source: [HuggingFace TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

### Training Phase (`--train`)

1. **Preprocessing Corpus**
   - GPT-4 style regex pre-tokenization
   - Streaming read/write (memory-efficient)
   - Saves to `data/processed/tiny_stories_train_processed.txt`

2. **Training BPE Tokenizer**
   - Byte-level base vocabulary (256 bytes)
   - Word frequency dict + incremental pair updates
   - Performs BPE merges until target vocab size
   - Default: 50,000 tokens (configurable)

3. **Saving Model**
   - `models/vocab.json` — Vocabulary mapping (id → token)
   - `models/merges.json` — BPE merge rules
   - `models/tokenizer_config.json` — Configuration

4. **Testing**
   - Runs test encoding/decoding on sample texts
   - Verifies round-trip correctness

## Output Files

After successful run:

```
models/
├── vocab.json              # Token ID to string mapping
├── merges.json             # BPE merge rules (pair → new_id)
└── tokenizer_config.json   # Tokenizer configuration

data/raw/
├── TinyStoriesV2-GPT4-train.txt  # Downloaded training data
└── TinyStoriesV2-GPT4-valid.txt  # Downloaded validation data

data/processed/
└── tiny_stories_train_processed.txt  # Preprocessed training data
```

## Using the Trained Tokenizer

```python
from src.bpe.tokenizer import GPT4Tokenizer

# Load trained tokenizer
tokenizer = GPT4Tokenizer()
tokenizer.load(
    vocab_path="models/vocab.json",
    merges_path="models/merges.json"
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
print(f"Vocabulary size: {tokenizer.vocab_size()}")
```

## Performance Tips

### Fast Training (for testing)

```bash
# Train on first 100MB only (much faster)
python src/setup_and_run.py --max-size 100 --vocab-size 10000
```

### Full Production Training

```bash
# Use full dataset with large vocabulary
python src/setup_and_run.py --vocab-size 50000
```

### Skip Preprocessing (if already done)

```bash
# Preprocess once
python src/setup_and_run.py --train

# Later, retrain with different vocab size (reuse preprocessed data)
python src/setup_and_run.py --train --no-preprocess --vocab-size 30000
```

## Troubleshooting

### "Training data not found"

Run setup first:
```bash
python src/setup_and_run.py --setup
```

### "Preprocessed data not found"

Either:
1. Run with preprocessing: `python src/setup_and_run.py --train`
2. Remove `--no-preprocess` flag

### Out of Memory

Limit corpus size:
```bash
python src/setup_and_run.py --max-size 100  # Use first 100MB only
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
- **RAM**: 4GB minimum, 8GB+ recommended for full dataset
- **Disk Space**: ~1GB for dataset + models
- **Internet**: Required for downloading dataset

## Dependencies

All dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `regex` — Unicode-aware pattern matching
- `requests` — Dataset downloading
- `tqdm` — Progress bars
- `tiktoken` — For benchmarking (optional)

## Architecture

### File Structure

```
src/
├── setup_and_run.py         # Main orchestration script
│   ├── setup()              # Create dirs, download data
│   └── run_training_pipeline()  # Preprocess, train, save
│
└── bpe/
    ├── tokenizer.py         # GPT4Tokenizer
    ├── train.py             # BPETrainer
    ├── encode_decode.py     # Encoder, Decoder
    ├── vocab.py             # Vocab
    ├── utils.py             # Preprocessing, streaming I/O
    ├── constants.py         # Paths, URLs
    └── download_data.py     # Dataset download
```

See [architecture_decision.md](architecture_decision.md) for design rationale.

## Next Steps

After training:

1. **Test tokenizer** — See examples above
2. **Benchmark** — `python benchmark.py`
3. **Try notebooks** — Explore `notebooks/` for interactive tutorials
4. **Custom data** — Modify corpus path in `setup_and_run.py`

## References

- [TinyStories Paper](https://arxiv.org/abs/2305.07759)
- [BPE Algorithm](https://arxiv.org/abs/1508.07909)
- [Streaming Usage](streaming_usage_example.md)
