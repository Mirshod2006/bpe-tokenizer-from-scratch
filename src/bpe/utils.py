"""Utility functions for BPE tokenizer operations."""

from collections import Counter, deque
from typing import List, Optional, Tuple, Union

def find_freq_pair(
    token_ids: List[int], 
    mode: str = "most"
) -> Optional[Tuple[int, int]]:
    """
    Find the most or least frequent adjacent pair in token IDs.
    
    Args:
        token_ids: List of token IDs
        mode: 'most' or 'least' frequent pair
    
    Returns:
        Tuple of (token_id1, token_id2) or None if no pairs exist
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
        raise ValueError("Invalid mode. Choose 'most' or 'least'.")


def replace_pair(
    token_ids: List[int], 
    pair_id: Tuple[int, int], 
    new_id: int
) -> List[int]:
    """
    Replace all occurrences of a pair with a new token ID.
    
    Args:
        token_ids: Original list of token IDs
        pair_id: Tuple of (token_id1, token_id2) to replace
        new_id: New token ID to replace the pair with
    
    Returns:
        List of token IDs with pair replaced
    """
    dq = deque(token_ids)
    replaced = []
    
    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair_id:
            replaced.append(new_id)
            dq.popleft()  # Remove the second token of the pair
        else:
            replaced.append(current)
    
    return replaced


def preprocess_text_for_gpt2(text: str) -> str:
    """
    Preprocess text for GPT-2 style tokenization.
    Replaces spaces with 'Ġ' except at the beginning.
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    processed = []
    for i, char in enumerate(text):
        if char == " " and i != 0:
            processed.append("Ġ")
        elif char != " ":
            processed.append(char)
    return "".join(processed)


def split_text_for_encoding(text: str) -> List[str]:
    """
    Split text into tokens for encoding (handles newlines and spaces).
    
    Args:
        text: Input text
    
    Returns:
        List of token strings
    """
    tokens = []
    lines = text.split("\n")
    
    for i, line in enumerate(lines):
        if i > 0:
            tokens.append("\n")
        
        words = line.split()
        for j, word in enumerate(words):
            if j == 0 and i > 0:
                tokens.append("Ġ" + word)
            elif j == 0:
                tokens.append(word)
            else:
                tokens.append("Ġ" + word)
    
    return tokens