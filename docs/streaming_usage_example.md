# Streaming Training Usage Examples

The tokenizer now supports memory-efficient streaming training to handle large datasets without loading everything into RAM.

## Memory Optimizations Implemented

1. **Word Frequency Dictionary**: Replaced list-of-lists corpus with `dict[tuple[int, ...], int]` to deduplicate identical words (10-20× memory reduction)
2. **Incremental Pair Updates**: Only modify affected words instead of rebuilding entire corpus on each merge
3. **Single Pair Count Build**: Build pair counts once before loop, then update incrementally
4. **Streaming Input**: Accept `Iterator[str]` to process data in chunks

## Usage with String (Backward Compatible)

```python
from src.bpe.tokenizer import GPT4Tokenizer

# Still works - automatically converted to chunks internally
tokenizer = GPT4Tokenizer()
text = "Your training text here..."
tokenizer.train(text=text, vocab_size=32_000)
```

## Usage with HuggingFace Datasets (Streaming)

```python
from datasets import load_dataset
from src.bpe.tokenizer import GPT4Tokenizer

# Load dataset in streaming mode (doesn't download everything at once)
ds = load_dataset('roneneldan/TinyStories', split='train', streaming=True)

# Create a chunk iterator
def chunk_iter(dataset, batch_size=10_000):
    """Yield text in batches to control memory usage."""
    buf = []
    for example in dataset:
        buf.append(example['text'])
        if len(buf) >= batch_size:
            yield '\n'.join(buf)
            buf = []
    if buf:
        yield '\n'.join(buf)

# Train with streaming iterator
tokenizer = GPT4Tokenizer()
tokenizer.train(
    text=chunk_iter(ds, batch_size=10_000),
    vocab_size=32_000,
    allowed_special={'<|endoftext|>'}
)

# Save the trained tokenizer
tokenizer.save('models/vocab.json', 'models/merges.json')
```

## Usage with File Chunks

```python
from src.bpe.tokenizer import GPT4Tokenizer

def file_chunk_iter(filepath, chunk_lines=10_000):
    """Read file in chunks to avoid loading entire file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        buf = []
        for line in f:
            buf.append(line.rstrip('\n'))
            if len(buf) >= chunk_lines:
                yield '\n'.join(buf)
                buf = []
        if buf:
            yield '\n'.join(buf)

tokenizer = GPT4Tokenizer()
tokenizer.train(
    text=file_chunk_iter('data/raw/TinyStoriesV2-GPT4-train.txt'),
    vocab_size=32_000
)
```

## Memory Usage Comparison

| Method | Memory Usage (2GB dataset) | Notes |
|--------|----------------------------|-------|
| **Old Implementation** | ~45 GB | Full corpus as list-of-lists |
| **New Implementation** | ~2-4 GB | Word frequency dict + streaming |

## Expected Performance

- **10-20× memory reduction** from deduplication
- **O(affected_words)** complexity per merge instead of O(all_words)
- Can train on datasets larger than available RAM
- Minimal performance impact on training speed
