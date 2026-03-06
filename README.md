# BPE Tokenizer from Scratch

A complete implementation of Byte Pair Encoding (BPE) tokenizer with comprehensive preprocessing filters and training pipeline.

## Features

- ✅ **Complete BPE Algorithm** - Training, encoding, and decoding
- ✅ **Text Preprocessing** - Unicode normalization, HTML decoding, URL handling, etc.
- ✅ **GPT-2 Pattern Matching** - Proper pre-tokenization with regex
- ✅ **TinyStories Dataset** - Automatic download and preprocessing
- ✅ **Easy Training** - One command to train from scratch
- ✅ **Model Persistence** - Save and load trained tokenizers

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test everything works
python test_setup.py

# Run full pipeline (download data + train tokenizer)
python setup&run.py

# Or step by step:
python setup&run.py --setup    # Download TinyStories dataset
python setup&run.py --train    # Train tokenizer
```

## Usage

### Training

```bash
# Default: 50K vocabulary
python setup&run.py

# Custom vocabulary size
python setup&run.py --vocab-size 30000

# Fast training (first 100MB only)
python setup&run.py --max-size 100
```

### Using Trained Tokenizer

```python
from src.bpe.tokenizer import BPETokenizer

# Load trained model
tokenizer = BPETokenizer()
tokenizer.load(
    vocab_path="src/models/vocab.json",
    merges_path="src/models/merges.json"
)

# Encode text
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
```

## Project Structure

```
├── setup&run.py              # Main pipeline (setup + train)
├── test_setup.py             # Quick verification tests
├── requirements.txt          # Python dependencies
│
├── src/bpe/
│   ├── tokenizer.py         # Main BPE tokenizer class
│   ├── trainer.py           # Training logic
│   ├── encoder.py           # Text → Token IDs
│   ├── decoder.py           # Token IDs → Text
│   ├── bpe_merger.py        # BPE merge operations
│   ├── utils.py             # Helper functions
│   └── constants.py         # Configuration
│
├── scripts/
│   ├── preprocess_corpus.py # Text preprocessing filters
│   ├── download_data.py     # Dataset download utilities
│   └── test_filters.py      # Filter demonstration
│
├── docs/
│   ├── setup_and_run_guide.md      # Complete usage guide
│   ├── preprocessing_filters.md    # Filter documentation
│   └── architecture_decision.md    # Design decisions
│
├── data/
│   ├── raw/                 # Downloaded datasets
│   └── processed/           # Preprocessed corpus
│
└── src/models/              # Trained tokenizer models
    ├── vocab.json
    ├── merges.json
    └── tokenizer_config.json
```

## Preprocessing Filters

The pipeline includes comprehensive text cleaning:

1. **Unicode Normalization** - Standardize character representations
2. **HTML Entity Decoding** - `&lt;` → `<`, `&amp;` → `&`
3. **URL/Email Normalization** - Replace with special tokens
4. **Control Character Removal** - Strip non-printable characters
5. **Whitespace Normalization** - Standardize spaces, tabs, newlines
6. **Pre-tokenization** - GPT-2 style regex pattern splitting

See [preprocessing_filters.md](docs/preprocessing_filters.md) for details.

## Documentation

- **[Setup & Run Guide](docs/setup_and_run_guide.md)** - Complete usage instructions
- **[Preprocessing Filters](docs/preprocessing_filters.md)** - Text cleaning pipeline
- **[Architecture Decision](docs/architecture_decision.md)** - Design rationale

## Requirements

- Python 3.7+
- regex (Unicode pattern matching)
- requests (dataset downloading)

```bash
pip install -r requirements.txt
```

## Testing

```bash
# Quick test (no dataset download)
python test_setup.py

# Test preprocessing filters
python scripts/test_filters.py
```

## Examples

### Fast Training (Testing)

```bash
# Train on first 100MB with small vocab (2-3 minutes)
python setup&run.py --max-size 100 --vocab-size 10000
```

### Production Training

```bash
# Full dataset with 50K vocab (30-60 minutes)
python setup&run.py --vocab-size 50000
```

### Retrain with Different Vocab Size

```bash
# Reuse preprocessed data, change vocab size only
python setup&run.py --train --no-preprocess --vocab-size 30000
```

## Architecture

**Design Choice**: Single `setup&run.py` file for the entire pipeline.

**Why?**
- Sequential operations (setup → preprocess → train → save)
- Shared configuration
- Single entry point
- Simple to understand and maintain

See [architecture_decision.md](docs/architecture_decision.md) for full rationale.

## Dataset

**TinyStories** - AI-generated short stories dataset

- Source: [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- Paper: [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- Train: ~500MB
- Validation: ~50MB

## License

[MIT License](LICENSE)

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- [OpenAI GPT-2](https://github.com/openai/gpt-2) - BPE implementation inspiration
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories) - Training data
- [Andrej Karpathy](https://github.com/karpathy) - Educational resources