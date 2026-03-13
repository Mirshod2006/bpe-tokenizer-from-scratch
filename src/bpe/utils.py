"""
utils.py — Static BPE utility functions.
No state; pure functions used by both training and encoding.
"""

from collections import Counter, deque
import os
import regex
from tqdm import tqdm
from typing import Optional, Iterator, Iterable
import logging
import sys

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|"""          # contractions
    r"""[^\r\n\p{L}\p{N}]?\p{L}+|"""       # words (optionally preceded by non-word)
    r"""\p{N}{1,3}|"""                      # numbers (up to 3 digits)
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*|"""       # punctuation / symbols
    r"""\s*[\r\n]+|"""                      # newlines
    r"""\s+(?!\S)|"""                       # trailing spaces
    r"""\s+"""                              # remaining whitespace
)

def logging_setup():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def count_pairs_in_corpus(corpus: list[list[int]]) -> Counter[tuple[int, int]]:
    """
    Count all adjacent pairs across all sequences in corpus efficiently.
    
    Args:
        corpus: List of token ID sequences.
    
    Returns:
        Counter of pair frequencies.
    """
    pair_counts: Counter[tuple[int, int]] = Counter()
    for seq in corpus:
        if len(seq) >= 2:
            pair_counts.update(zip(seq, seq[1:]))
    return pair_counts



def replace_pair(token_ids: list[int], pair_id: tuple[int, int], new_id: int) -> list[int]:
    """
    Replace all occurrences of pair_id in token_ids with new_id.

    Args:
        token_ids: Original token ID list.
        pair_id: The (left, right) pair to replace.
        new_id: The merged token ID to insert.

    Returns:
        New token ID list with replacements applied.
    """
    dq: deque[int] = deque(token_ids)
    replaced: list[int] = []

    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair_id:
            replaced.append(new_id)
            dq.popleft()  # consume the second token of the pair
        else:
            replaced.append(current)

    return replaced


def preprocess_text_gpt4(text: str) -> Iterator[str]:   # takes a string
    for match in regex.finditer(GPT4_SPLIT_PATTERN, text):
        yield match.group()

def preprocess_chunks(chunks: Iterable[str]) -> Iterator[str]:  # takes an iterator
    for chunk in chunks:
        yield from preprocess_text_gpt4(chunk)

def read_corpus_in_chunks(file_path: str, chunk_size: int = 10_000) -> Iterator[str]:
    """
    Lazily yield text chunks from a file, chunk_size lines at a time.
    Never holds more than one chunk in memory.
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        chunk_lines: list[str] = []
        for line in f:
            chunk_lines.append(line)
            if len(chunk_lines) >= chunk_size:
                yield ''.join(chunk_lines)
                chunk_lines.clear()          # release memory immediately
        if chunk_lines:
            yield ''.join(chunk_lines)

def write_corpus_in_chunks(
    file_path: str,
    text_iter: Iterable[str],
    write_batch: int = 5_000,       # flush to disk every N tokens
    separator: str = "\n",
) -> int:
    """
    Write tokens from a lazy iterator to a file in batches.

    Args:
        file_path:   Destination path.
        text_iter:   Any iterator yielding strings (tokens, lines, chunks).
        write_batch: Number of tokens to buffer before each disk write.
        separator:   String placed between tokens in output file.

    Returns:
        Total number of tokens written.
    """
    total = 0
    buffer: list[str] = []

    with open(file_path, 'w', encoding='utf-8') as f:
        for token in text_iter:
            buffer.append(token)
            if len(buffer) >= write_batch:
                f.write(separator.join(buffer) + separator)
                buffer.clear()
                total += write_batch

        if buffer:                           # flush remainder
            f.write(separator.join(buffer))
            total += len(buffer)

    return total

def preprocess_corpus(
    input_path: str,
    output_path: str,
    max_size_mb: Optional[int] = 200,
    chunk_size: int = 10_000,
) -> str:
    """
    Stream-process a corpus file through the full read → tokenize → write
    pipeline without ever loading the full file into memory.

    Memory profile: O(chunk_size lines) at any point in time.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    logging.info(f"Preprocessing: {input_path}  ({file_size_mb:.2f} MB)")

    if max_size_mb and file_size_mb > max_size_mb:
        logging.warning(f"File exceeds {max_size_mb} MB limit — continuing anyway.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    logging.info('Applying GPT-4 tokenization and writing to: ' + output_path)

    # ── The entire pipeline is lazy — nothing is evaluated until write consumes it
    chunks    = read_corpus_in_chunks(input_path, chunk_size=chunk_size)
    tokens    = preprocess_chunks(tqdm(chunks, desc="Reading corpus", unit=" chunks"))          # generator, not a list
    n_written = write_corpus_in_chunks(output_path, tqdm(tokens, desc="Writing", unit=" tokens"))

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"✓ Written {n_written:,} tokens → {output_path}  ({output_size_mb:.2f} MB)")

    return output_path


def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2/GPT-4 byte-level BPE mapping: maps raw bytes (0-255) to
    printable unicode characters so every byte has a visible representation.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) +
        list(range(ord("¡"), ord("¬") + 1)) +
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}