# BPE Tokenizer from Scratch

A complete implementation of Byte Pair Encoding (BPE) tokenizer with GPT-4 style architecture. Train from scratch on your own data or load OpenAI's pretrained vocab and merges.

## Features

- **GPT-4 Style BPE** — Byte-level BPE with GPT-4 pre-tokenization pattern
- **Training** — Train from scratch on raw text with memory-efficient streaming
- **OpenAI Compatible** — Load pretrained `cl100k_base` / GPT-2 vocab and merges
- **Encode / Decode** — Full text ↔ token ID conversion with special token support
- **TinyStories Dataset** — Automatic download and preprocessing
- **Model Persistence** — Save and load custom trained tokenizers
- **Benchmarking** — Compare with tiktoken (GPT-2, GPT-3, GPT-4, GPT-4o)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test everything works
python tests/test_setup.py

# Run full pipeline (download data + train tokenizer)
python src/setup_and_run.py

# Or step by step:
python src/setup_and_run.py --setup    # Download TinyStories dataset
python src/setup_and_run.py --train    # Train tokenizer
```

## Usage

### Training

```bash
# Default: 50K vocabulary
python src/setup_and_run.py

# Custom vocabulary size
python src/setup_and_run.py --vocab-size 30000

# Fast training (first 100MB only)
python src/setup_and_run.py --max-size 100

# Skip preprocessing (use existing processed data)
python src/setup_and_run.py --train --no-preprocess --vocab-size 30000
```

### Using Trained Tokenizer

```python
from src.bpe.tokenizer import GPT4Tokenizer

# Load trained model
tokenizer = GPT4Tokenizer()
tokenizer.load(
    vocab_path="models/vocab.json",
    merges_path="models/merges.json"
)

# Encode text
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Vocabulary size
print(f"Vocab size: {tokenizer.vocab_size()}")
```

### Loading OpenAI Pretrained

```python
from src.bpe.tokenizer import GPT4Tokenizer

tokenizer = GPT4Tokenizer()
tokenizer.load_from_openai(
    vocab_path="path/to/encoder.json",
    merges_path="path/to/vocab.bpe"
)
tokens = tokenizer.encode("Hello, world!")
```

## Project Structure

```
├── benchmark.py              # Benchmark custom tokenizer vs tiktoken
├── requirements.txt          # Python dependencies
├── tests/
│   └── test_setup.py         # Quick verification tests
│
├── src/
│   ├── setup_and_run.py      # Main pipeline (setup + train + save)
│   └── bpe/
│       ├── tokenizer.py      # GPT4Tokenizer - main class
│       ├── train.py          # BPETrainer - BPE training logic
│       ├── encode_decode.py  # Encoder, Decoder - text ↔ token IDs
│       ├── vocab.py          # Vocab - token/ID mappings
│       ├── utils.py          # Preprocessing, bytes_to_unicode, streaming
│       ├── constants.py      # Paths, special tokens, dataset URLs
│       └── download_data.py  # Dataset download (TinyStories, WikiText, FineWeb)
│
├── notebooks/                # Interactive tutorials
│   ├── bpe_algorithm_explanation.ipynb  # BPE algorithm walkthrough
│   ├── karpathy_tokenizer_exp.ipynb     # Karpathy-style tokenization exploration
│   ├── tokenization_examples.ipynb      # Comparison & edge cases
│   └── training_visualization.ipynb     # Visualize merge process
│
├── docs/
│   ├── setup_and_run_guide.md    # Complete usage instructions
│   ├── streaming_usage_example.md # Memory-efficient training
│   └── architecture_decision.md  # Design rationale
│
├── data/
│   ├── raw/                  # Downloaded datasets
│   └── processed/            # Preprocessed corpus
│
└── models/                   # Trained tokenizer (created after training)
    ├── vocab.json
    ├── merges.json
    ├── tokenizer_config.json
    └── benchmark_results.json
```

## Source Code Overview

| Module | Purpose |
|--------|---------|
| `tokenizer.py` | `GPT4Tokenizer` — orchestrates Vocab, BPETrainer, Encoder, Decoder |
| `train.py` | `BPETrainer` — byte-level BPE merges, incremental pair updates |
| `encode_decode.py` | `Encoder` / `Decoder` — text → IDs, IDs → text (byte-level + merges) |
| `vocab.py` | `Vocab` — ID↔token mapping, OpenAI format loading |
| `utils.py` | `preprocess_text_gpt4`, `bytes_to_unicode`, streaming corpus I/O |
| `constants.py` | Paths, special tokens, dataset URLs |

## Pre-tokenization

GPT-4 style regex pattern handles:
- Contractions (`'s`, `'t`, `'re`, etc.)
- Words (Unicode letters)
- Numbers (up to 3 digits)
- Punctuation and symbols
- Whitespace and newlines

## Notebooks

| Notebook | Description |
|----------|-------------|
| `bpe_algorithm_explanation.ipynb` | Step-by-step BPE algorithm with visualizations |
| `karpathy_tokenizer_exp.ipynb` | Karpathy-style exploration: Unicode, bytes, merges, GPT-2 patterns, SentencePiece |
| `tokenization_examples.ipynb` | Tokenization comparisons, TinyStories samples, edge cases |
| `training_visualization.ipynb` | Visualize vocabulary evolution and merge statistics |

## Documentation

- **[Setup & Run Guide](docs/setup_and_run_guide.md)** — CLI options, troubleshooting
- **[Streaming Usage](docs/streaming_usage_example.md)** — Memory-efficient training
- **[Notebooks Overview](docs/notebooks_overview.md)** — Interactive tutorials
- **[Architecture Decision](docs/architecture_decision.md)** — Design rationale

## Benchmarking

Compare the custom tokenizer with tiktoken models (GPT-2, GPT-3, GPT-4, GPT-4o):

```bash
# Train first, then benchmark
python src/setup_and_run.py --max-size 100 --vocab-size 10000
python benchmark.py

# Custom paths
python benchmark.py --vocab models/vocab.json --merges models/merges.json
python benchmark.py --no-tiktoken  # Skip tiktoken, benchmark custom only
```

## Requirements

- Python 3.7+
- `regex` — Unicode pattern matching (GPT-4 pre-tokenization)
- `requests` — Dataset downloading
- `tqdm` — Progress bars
- `tiktoken` — For benchmarking (optional)

```bash
pip install -r requirements.txt
```

## Dataset

**TinyStories** — AI-generated short stories for training

- Source: [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories)
- Paper: [TinyStories: How Small Can Language Models Be?](https://arxiv.org/abs/2305.07759)
- Train: ~500MB, Validation: ~50MB

Additional datasets supported via `download_data.py`: WikiText-103, FineWeb.

## License

[MIT License](LICENSE)

## Acknowledgments

- [OpenAI GPT-2/GPT-4](https://github.com/openai/gpt-2) — BPE implementation inspiration
- [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories) — Training data
- [Andrej Karpathy](https://github.com/karpathy) — Educational resources on tokenization
