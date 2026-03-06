"""BPE encoding module."""

import re
from typing import List, Optional, Set, Union
from functools import lru_cache

from .constants import GPT2_SPACE_TOKEN, NEWLINE_TOKEN
from .utils import split_text_for_encoding


class BPEEncoder:
    """
    Handles text encoding to token IDs.
    Supports both custom and GPT-2 style tokenization.
    """
    
    def __init__(self, vocab: dict[int, str], inverse_vocab: dict[str, int], merger):
        """
        Initialize encoder with vocabulary and merger.
        
        Args:
            vocab: Token ID to token string mapping
            inverse_vocab: Token string to token ID mapping
            merger: BPEMerger instance for BPE operations
        """
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.merger = merger
    
    def encode(
        self, 
        text: str, 
        allowed_special: Optional[Set[str]] = None
    ) -> List[int]:
        """
        Encode text to token IDs with special token handling.
        
        Args:
            text: Input text
            allowed_special: Set of allowed special tokens
        
        Returns:
            List of token IDs
        """
        token_ids = []
        
        # Handle special tokens if enabled
        if allowed_special:
            token_ids = self._encode_with_special_tokens(text, allowed_special)
        else:
            token_ids = self._encode_regular(text)
        
        return token_ids
    
    def _encode_with_special_tokens(self, text: str, allowed_special: Set[str]) -> List[int]:
        """Encode text with special token handling."""
        token_ids = []
        
        # Build regex for special tokens
        special_pattern = "(" + "|".join(
            re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)
        ) + ")"
        
        last_index = 0
        for match in re.finditer(special_pattern, text):
            # Encode prefix without special handling
            prefix = text[last_index:match.start()]
            if prefix:
                token_ids.extend(self._encode_regular(prefix))
            
            # Handle special token
            special_token = match.group(0)
            if special_token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[special_token])
            else:
                raise ValueError(f"Special token '{special_token}' not found in vocabulary")
            
            last_index = match.end()
        
        # Encode remaining text
        remaining = text[last_index:]
        if remaining:
            token_ids.extend(self._encode_regular(remaining))
        
        # Check for disallowed special tokens in remaining text
        self._check_disallowed_special_tokens(remaining, allowed_special)
        
        return token_ids
    
    def _encode_regular(self, text: str) -> List[int]:
        """Encode regular text without special token handling."""
        token_ids = []
        tokens = split_text_for_encoding(text)
        
        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                # Tokenize unknown token with BPE
                token_ids.extend(self._tokenize_with_bpe(token))
        
        return token_ids
    
    def _tokenize_with_bpe(self, token: str) -> List[int]:
        """
        Tokenize a single token using BPE merges.
        
        Args:
            token: Token string to tokenize
        
        Returns:
            List of token IDs
        """
        # Convert token to character IDs
        token_ids = self._token_to_char_ids(token)
        
        # Apply custom merges if available
        if self.merger.bpe_merges:
            return self.merger.merge_tokens(token_ids)
        
        # Apply GPT-2 style ranked merges
        symbols = [self.vocab[id_num] for id_num in token_ids]
        merged_symbols = self.merger.merge_with_ranks(symbols)
        
        return [self.inverse_vocab[sym] for sym in merged_symbols]
    
    def _token_to_char_ids(self, token: str) -> List[int]:
        """Convert token string to list of character token IDs."""
        token_ids = []
        missing_chars = []
        
        for char in token:
            token_id = self.inverse_vocab.get(char)
            if token_id is None:
                missing_chars.append(char)
            else:
                token_ids.append(token_id)
        
        if missing_chars:
            raise ValueError(f"Characters not found in vocab: {missing_chars}")
        
        return token_ids
    
    def _check_disallowed_special_tokens(self, text: str, allowed_special: Set[str]):
        """Check for disallowed special tokens in text."""
        disallowed = [
            tok for tok in self.inverse_vocab
            if tok.startswith("<|") and tok.endswith("|>") 
            and tok in text and tok not in allowed_special
        ]
        if disallowed:
            raise ValueError(f"Disallowed special tokens encountered: {disallowed}")
    
    @lru_cache(maxsize=1000)
    def get_token_id(self, token: str) -> Optional[int]:
        """Get token ID for a token string (cached)."""
        return self.inverse_vocab.get(token)