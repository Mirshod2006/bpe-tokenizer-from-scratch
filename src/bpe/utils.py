"""
utils.py — Static BPE utility functions.
No state; pure functions used by both training and encoding.
"""

from collections import Counter, deque
import os
from typing import Optional


def find_freq_pair(token_ids: list[int], mode: str = "most") -> tuple[int, int] | None:
    """
    Find the most or least frequent adjacent pair in a token ID sequence.

    Args:
        token_ids: List of token IDs.
        mode: 'most' or 'least'.

    Returns:
        The best pair tuple, or None if no pairs exist.
    """
    if len(token_ids) < 2:
        return None

    pairs = Counter(zip(token_ids, token_ids[1:]))
    if not pairs:
        return None

    if mode == "most":
        return max(pairs.items(), key=lambda x: x[1])[0]
    elif mode == "least":
        return min(pairs.items(), key=lambda x: x[1])[0]
    else:
        raise ValueError("mode must be 'most' or 'least'.")


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


def preprocess_text_gpt4(text: str) -> list[str]:
    """
    GPT-4 (cl100k_base) style pre-tokenization using regex splitting.
    Preserves spaces as part of tokens (no Ġ substitution).

    Returns a list of string "words" to be BPE-encoded individually.
    """
    import regex as re  # requires: pip install regex

    # GPT-4 cl100k_base split pattern (from tiktoken source)
    GPT4_SPLIT_PATTERN = (
        r"""'(?i:[sdmt]|ll|ve|re)|"""          # contractions
        r"""[^\r\n\p{L}\p{N}]?\p{L}+|"""       # words (optionally preceded by non-word)
        r"""\p{N}{1,3}|"""                      # numbers (up to 3 digits)
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*|"""       # punctuation / symbols
        r"""\s*[\r\n]+|"""                      # newlines
        r"""\s+(?!\S)|"""                       # trailing spaces
        r"""\s+"""                              # remaining whitespace
    )
    return re.findall(GPT4_SPLIT_PATTERN, text)

def preprocess_corpus(
    input_path: str,
    output_path: str,
    max_size_mb: Optional[int] = None
) -> str:
    """
    Preprocess corpus file using the filtering pipeline.
    
    Args:
        input_path: Path to raw corpus file
        output_path: Path to save preprocessed corpus
        max_size_mb: Optional limit on input file size (in MB)
    
    Returns:
        Path to preprocessed file
    """
    print(f"\n📝 Preprocessing: {input_path}")
    
    # Check file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check file size
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    
    if max_size_mb and file_size_mb > max_size_mb:
        print(f"   ⚠ File exceeds {max_size_mb} MB limit. Processing first {max_size_mb} MB only.")
    
    # Read and preprocess
    print("   Applying filters...")
    with open(input_path, 'r', encoding='utf-8') as f:
        # Read limited size if specified
        if max_size_mb:
            max_bytes = max_size_mb * 1024 * 1024
            text = f.read(max_bytes)
        else:
            text = f.read()
    
    # Apply preprocessing pipeline
    cleaned_text = preprocess_text_gpt4(
        text
    )
    
    # Save preprocessed text
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(cleaned_text))
    
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   ✓ Saved preprocessed text: {output_path}")
    print(f"   Output size: {output_size_mb:.2f} MB")
    
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