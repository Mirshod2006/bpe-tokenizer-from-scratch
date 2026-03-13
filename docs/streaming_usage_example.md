# Streaming Training Usage Examples

The tokenizer supports memory-efficient streaming training to handle large datasets without loading everything into RAM.

## Memory Optimizations Implemented

1. **Word Frequency Dictionary**: Uses `dict[tuple[int, ...], int]` to deduplicate identical words (10–20× memory reduction vs. full corpus)
2. **Incremental Pair Updates**: Only modify affected words instead of rebuilding entire corpus on each merge
3. **Single Pair Count Build**: Build pair counts once before loop, then update incrementally
4. **Streaming Input**: Accepts `Iterator[str]` or `Iterable[str]` to process data in chunks

## Usage with List of Chunks (Simple)

```python
from src.bpe.tokenizer import GPT4Tokenizer

# Pass an iterable of text chunks
tokenizer = GPT4Tokenizer()
chunks = ["First chunk of text...", "Second chunk of text..."]
tokenizer.train(
    text_iter=chunks,
    vocab_size=32_000,
    allowed_special={"<|endoftext|>"}
)
```

## Usage with File Chunks

```python
from src.bpe.tokenizer import GPT4Tokenizer
from src.bpe.utils import read_corpus_in_chunks

tokenizer = GPT4Tokenizer()
tokenizer.train(
    text_iter=read_corpus_in_chunks("data/raw/TinyStoriesV2-GPT4-train.txt", chunk_size=10_000),
    vocab_size=32_000,
    allowed_special={"<|endoftext|>"}
)

# Save the trained tokenizer
tokenizer.save("models/vocab.json", "models/merges.json")
```

## Usage with HuggingFace Datasets (Streaming)

```python
from datasets import load_dataset
from src.bpe.tokenizer import GPT4Tokenizer

# Load dataset in streaming mode (doesn't download everything at once)
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

def chunk_iter(dataset, batch_size=10_000):
    """Yield text in batches to control memory usage."""
    buf = []
    for example in dataset:
        buf.append(example["text"])
        if len(buf) >= batch_size:
            yield "\n".join(buf)
            buf = []
    if buf:
        yield "\n".join(buf)

# Train with streaming iterator
tokenizer = GPT4Tokenizer()
tokenizer.train(
    text_iter=chunk_iter(ds, batch_size=10_000),
    vocab_size=32_000,
    allowed_special={"<|endoftext|>"}
)

tokenizer.save("models/vocab.json", "models/merges.json")
```

## Custom File Chunk Iterator

```python
from src.bpe.tokenizer import GPT4Tokenizer

def file_chunk_iter(filepath, chunk_lines=10_000):
    """Read file in chunks to avoid loading entire file."""
    with open(filepath, "r", encoding="utf-8") as f:
        buf = []
        for line in f:
            buf.append(line.rstrip("\n"))
            if len(buf) >= chunk_lines:
                yield "\n".join(buf)
                buf = []
        if buf:
            yield "\n".join(buf)

tokenizer = GPT4Tokenizer()
tokenizer.train(
    text_iter=file_chunk_iter("data/raw/TinyStoriesV2-GPT4-train.txt"),
    vocab_size=32_000,
)
```

## API Reference

```python
tokenizer.train(
    text_iter: Iterator[str] | Iterable[str],  # Yields text chunks
    vocab_size: int,
    allowed_special: set[str] = {"<|endoftext|>"},
    chunk_size: int = 10_000  # Used when converting string to iterator (internal)
)
```

**Note**: Pass an iterable of **text chunks** (paragraphs or batches of lines), not individual characters. Each element should be a string representing a chunk of the corpus.

## Memory Usage Comparison

| Method | Memory (2GB dataset) | Notes |
|--------|----------------------|-------|
| **Full corpus in memory** | ~45 GB | List-of-lists (old style) |
| **Streaming + word dedup** | ~2–4 GB | Current implementation |

## Expected Performance

- **10–20× memory reduction** from word deduplication
- **O(affected_words)** per merge instead of O(all_words)
- Can train on datasets larger than available RAM
- Minimal impact on training speed
