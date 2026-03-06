"""BPE tokenizer training module."""

from typing import List, Set, Optional
from collections import Counter

from .constants import GPT2_SPACE_TOKEN, DEFAULT_SPECIAL_TOKENS
from .utils import preprocess_text_for_gpt2, find_freq_pair, replace_pair


class BPETrainer:
    """
    Handles BPE training logic.
    """
    
    def __init__(self):
        self.vocab: dict[int, str] = {}
        self.inverse_vocab: dict[str, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
    
    def train(
        self, 
        text: str, 
        vocab_size: int,
        allowed_special: Optional[Set[str]] = None
    ) -> tuple[dict[int, str], dict[str, int], dict[tuple[int, int], int]]:
        """
        Train BPE tokenizer from scratch.
        
        Args:
            text: Training text
            vocab_size: Desired vocabulary size
            allowed_special: Set of special tokens to include
        
        Returns:
            Tuple of (vocab, inverse_vocab, merges)
        """
        if allowed_special is None:
            allowed_special = DEFAULT_SPECIAL_TOKENS
        
        # Preprocess text
        processed_text = preprocess_text_for_gpt2(text)
        
        # Initialize vocabulary with characters
        self._initialize_vocab(processed_text, allowed_special)
        
        # Tokenize the processed text
        token_ids = [self.inverse_vocab[char] for char in processed_text]
        
        # Perform BPE merges
        self._perform_merges(token_ids, vocab_size)
        
        # Build final vocabulary with merged tokens
        self._build_merged_vocab()
        
        return self.vocab, self.inverse_vocab, self.merges
    
    def _initialize_vocab(self, text: str, special_tokens: Set[str]):
        """Initialize vocabulary with characters and special tokens."""
        # Start with first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        
        # Add any additional characters from text
        unique_chars.extend(
            char for char in sorted(set(text))
            if char not in unique_chars
        )
        
        # Ensure space token is included
        if GPT2_SPACE_TOKEN not in unique_chars:
            unique_chars.append(GPT2_SPACE_TOKEN)
        
        # Build vocab
        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}
        
        # Add special tokens
        for token in special_tokens:
            if token not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.inverse_vocab[token] = new_id
    
    def _perform_merges(self, token_ids: List[int], target_vocab_size: int):
        """Perform BPE merges until target vocabulary size is reached."""
        current_vocab_size = len(self.vocab)
        
        for new_id in range(current_vocab_size, target_vocab_size):
            pair_id = find_freq_pair(token_ids, mode="most")
            
            if pair_id is None:
                break
            
            token_ids = replace_pair(token_ids, pair_id, new_id)
            self.merges[pair_id] = new_id
    
    def _build_merged_vocab(self):
        """Build final vocabulary including merged tokens."""
        for (p0, p1), new_id in self.merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id