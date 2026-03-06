import re
import os
import sys
import html
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict

from src.bpe.constants import GPT2_PATTERN
from src.bpe.bpe_merger import BPEMerger


# ============================================================================
# TEXT CLEANING FILTERS (Applied in order)
# ============================================================================

def normalize_unicode(text: str, form: str = 'NFC') -> str:
    """
    Step 1: Unicode Normalization
    
    Normalize Unicode to handle different representations of the same character.
    - NFC (Canonical Composition): Recommended for most cases
    - NFKC (Compatibility Composition): More aggressive, converts variants
    
    Args:
        text: Input text
        form: Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
    
    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)


def decode_html_entities(text: str) -> str:
    """
    Step 2: HTML Entity Decoding
    
    Convert HTML entities to their corresponding characters.
    Example: '&amp;' -> '&', '&lt;' -> '<', '&#39;' -> "'"
    
    Args:
        text: Input text with potential HTML entities
    
    Returns:
        Text with decoded HTML entities
    """
    return html.unescape(text)


def normalize_urls_and_emails(text: str, replace_with: str = ' <URL> ') -> str:
    """
    Step 3: URL and Email Normalization
    
    Replace URLs and email addresses with special tokens to:
    - Reduce vocabulary size
    - Prevent learning domain-specific patterns
    - Maintain semantic meaning
    
    Args:
        text: Input text
        replace_with: Token to replace URLs/emails with (use None to keep original)
    
    Returns:
        Text with normalized URLs/emails
    """
    if replace_with is None:
        return text
    
    # URL pattern (http, https, ftp, www)
    url_pattern = r'(?:http[s]?://|ftp://|www\.)[^\s]+'
    text = re.sub(url_pattern, replace_with, text)
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, ' <EMAIL> ', text)
    
    return text


def remove_control_characters(text: str, keep_chars: set = set()) -> str:
    """
    Step 4: Control Character Removal
    
    Remove non-printable control characters while preserving:
    - Newlines (\n)
    - Tabs (\t)
    - Carriage returns (\r)
    
    Args:
        text: Input text
        keep_chars: Set of characters to preserve (default: {'\n', '\t', '\r'})
    
    Returns:
        Text with control characters removed
    """
    if keep_chars is None:
        keep_chars = {'\n', '\t', '\r'}
    
    cleaned = []
    for char in text:
        # Keep printable characters and explicitly allowed control chars
        if char in keep_chars or unicodedata.category(char)[0] != 'C':
            cleaned.append(char)
    
    return ''.join(cleaned)


def normalize_whitespace(text: str) -> str:
    """
    Step 5: Whitespace Normalization
    
    Standardize all whitespace:
    - Convert tabs to spaces
    - Replace multiple spaces with single space
    - Preserve single newlines
    - Remove multiple consecutive newlines (reduce to max 2)
    
    Args:
        text: Input text
    
    Returns:
        Text with normalized whitespace
    """
    # Convert tabs to spaces
    text = text.replace('\t', ' ')
    
    # Replace carriage returns with newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace multiple spaces with single space (but not across newlines)
    text = re.sub(r' +', ' ', text)
    
    # Reduce multiple newlines to maximum of 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove spaces at beginning/end of lines
    text = re.sub(r' *\n *', '\n', text)
    
    return text.strip()


# ============================================================================
# PRE-TOKENIZATION FILTERS (Regex-based splitting)
# ============================================================================

# def pretokenize_with_gpt2_pattern(text: str) -> List[str]:
#     """
#     Step 6: Pre-tokenization with Regex Pattern
    
#     Split text using GPT2-style pattern to prevent merging across:
#     - Whitespace boundaries
#     - Word boundaries
#     - Punctuation boundaries
#     - Number grouping
    
#     Pattern handles:
#     - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
#     - Letters: optional space + sequence of letters
#     - Numbers: optional space + sequence of numbers
#     - Other: optional space + non-letter/non-number characters
#     - Whitespace sequences
    
#     Args:
#         text: Input text
    
#     Returns:
#         List of pre-tokenized chunks
#     """
#     # Use the GPT2 pattern for splitting (not removing!)
#     import regex  # GPT2_PATTERN uses \p{L} which requires regex module
    
#     # Using findall instead of sub to extract matching patterns
#     tokens = regex.findall(GPT2_PATTERN, text)
    
#     # Filter out empty tokens
#     return [token for token in tokens if token.strip()]


# def pretokenize_simple(text: str) -> List[str]:
#     """
#     Alternative: Simple Pre-tokenization (fallback if regex module unavailable)
    
#     Basic splitting on whitespace and punctuation boundaries.
    
#     Args:
#         text: Input text
    
#     Returns:
#         List of pre-tokenized chunks
#     """
#     # Split on whitespace but keep words intact
#     # This is a simpler alternative if the regex module is not available
#     pattern = r"\w+(?:'\w+)?|[^\w\s]|\s+"
#     tokens = re.findall(pattern, text)
#     return [token for token in tokens if token.strip()]


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_corpus(
    file_path: str,
    normalize_urls: bool = True,
    unicode_form: str = 'NFC'
) -> str:
    """
    Complete preprocessing pipeline for corpus text.
    
    Pipeline order:
    1. Unicode normalization
    2. HTML entity decoding
    3. URL/email normalization
    4. Control character removal
    5. Whitespace normalization
    6. Pre-tokenization (regex-based splitting)
    
    Args:
        file_path: Path to corpus file
        normalize_urls: Whether to replace URLs/emails with tokens
        unicode_form: Unicode normalization form ('NFC', 'NFKC', etc.)
        use_gpt2_pattern: Use GPT2 pattern (requires regex module) or simple pattern
    
    Returns:
        List of pre-tokenized text chunks
    """
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Apply filters in order
    text = normalize_unicode(text, form=unicode_form)
    text = decode_html_entities(text)
    
    if normalize_urls:
        text = normalize_urls_and_emails(text, replace_with=' <URL> ')
    
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    
    # # Pre-tokenization
    # if use_gpt2_pattern:
    #     try:
    #         tokens = pretokenize_with_gpt2_pattern(text)
    #     except (ImportError, Exception) as e:
    #         print(f"Warning: Could not use GPT2 pattern ({e}). Falling back to simple tokenization.")
    #         tokens = pretokenize_simple(text)
    # else:
    #     tokens = pretokenize_simple(text)
    
    return text


def preprocess_text_string(
    text: str,
    normalize_urls: bool = True,
    unicode_form: str = 'NFC'
) -> str:
    """
    Preprocess a text string (without pre-tokenization splitting).
    
    Useful for cleaning text while keeping it as a single string.
    
    Args:
        text: Input text string
        normalize_urls: Whether to replace URLs/emails with tokens
        unicode_form: Unicode normalization form
    
    Returns:
        Cleaned text string
    """
    text = normalize_unicode(text, form=unicode_form)
    text = decode_html_entities(text)
    
    if normalize_urls:
        text = normalize_urls_and_emails(text, replace_with=' <URL> ')
    
    text = remove_control_characters(text)
    text = normalize_whitespace(text)
    
    return text

