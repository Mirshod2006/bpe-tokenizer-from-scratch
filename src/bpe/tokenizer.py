"""Main BPE Tokenizer class that orchestrates all components."""

import json
from typing import List, Optional, Set, Union

from .constants import DEFAULT_SPECIAL_TOKENS, GPT2_SPACE_TOKEN
from .trainer import BPETrainer
from .encoder import BPEEncoder
from .decoder import BPEDecoder
from .bpe_merger import BPEMerger
from . import utils


class BPETokenizer:
    """
    Main BPE Tokenizer class that orchestrates training, encoding, and decoding.
    Supports both custom training and loading pre-trained GPT-2 tokenizers.
    """
    
    def __init__(self):
        # Core data structures
        self.vocab: dict[int, str] = {}
        self.inverse_vocab: dict[str, int] = {}
        
        # Components
        self.merger = BPEMerger()
        self.encoder: Optional[BPEEncoder] = None
        self.decoder: Optional[BPEDecoder] = None
        
        # Initialize components after vocab is set
        self._initialized = False
    
    def _initialize_components(self):
        """Initialize encoder and decoder with current vocab."""
        self.merger.set_vocab_references(self.vocab, self.inverse_vocab)
        self.encoder = BPEEncoder(self.vocab, self.inverse_vocab, self.merger)
        self.decoder = BPEDecoder(self.vocab)
        self._initialized = True
    
    # ==================== Training ====================
    
    def train(
        self, 
        text: str, 
        vocab_size: int,
        allowed_special: Optional[Set[str]] = None
    ):
        """
        Train the BPE tokenizer from scratch.
        
        Args:
            text: Training text
            vocab_size: Desired vocabulary size
            allowed_special: Set of special tokens to include
        """
        trainer = BPETrainer()
        self.vocab, self.inverse_vocab, self.merger.bpe_merges = trainer.train(
            text, vocab_size, allowed_special
        )
        self._initialize_components()
    
    # ==================== Loading Pre-trained ====================
    
    def load_vocab_and_merges_from_openai(self, vocab_path: str, bpe_merges_path: str):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.
        
        Args:
            vocab_path: Path to encoder.json
            bpe_merges_path: Path to vocab.bpe
        """
        # Load vocabulary
        self._load_openai_vocab(vocab_path)
        
        # Load merges
        self._load_openai_merges(bpe_merges_path)
        
        self._initialize_components()
    
    def _load_openai_vocab(self, vocab_path: str):
        """Load OpenAI vocabulary file."""
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}
        
        # Handle newline character
        self._ensure_newline_in_vocab()
    
    def _ensure_newline_in_vocab(self):
        """Ensure newline token is in vocabulary."""
        if "\n" in self.inverse_vocab:
            return
        
        # Find a suitable fallback token
        fallback_token = next(
            (token for token in ["<|endoftext|>", "Ġ", ""] 
             if token in self.inverse_vocab),
            None
        )
        
        if fallback_token is not None:
            newline_id = self.inverse_vocab[fallback_token]
            self.inverse_vocab["\n"] = newline_id
            self.vocab[newline_id] = "\n"
        else:
            raise KeyError("No suitable token found to map '\\n'")
    
    def _load_openai_merges(self, merges_path: str):
        """Load OpenAI merges file."""
        self.merger.bpe_ranks = {}
        
        with open(merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            
            # Skip header if present
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
            
            for rank, line in enumerate(lines):
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    # Only add if both tokens are in vocabulary
                    if (token1 in self.inverse_vocab and 
                        token2 in self.inverse_vocab):
                        self.merger.bpe_ranks[pair] = rank
    
    # ==================== Saving/Loading Custom ====================
    
    def save(self, vocab_path: str, merges_path: str):
        """
        Save vocabulary and merges to files.
        
        Args:
            vocab_path: Path to save vocabulary JSON
            merges_path: Path to save merges JSON
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)
        
        # Save merges
        with open(merges_path, "w", encoding="utf-8") as file:
            json.dump(self.merger.to_dict(), file, ensure_ascii=False, indent=2)
    
    def load(self, vocab_path: str, merges_path: str):
        """
        Load vocabulary and merges from files.
        
        Args:
            vocab_path: Path to vocabulary JSON
            merges_path: Path to merges JSON
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}
        
        # Load merges
        with open(merges_path, "r", encoding="utf-8") as file:
            merges_data = json.load(file)
            self.merger = BPEMerger.from_dict(merges_data)
        
        self._initialize_components()
    
    # ==================== Encoding/Decoding ====================
    
    def encode(
        self, 
        text: str, 
        allowed_special: Optional[Set[str]] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            allowed_special: Set of allowed special tokens
        
        Returns:
            List of token IDs
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized. Train or load first.")
        
        return self.encoder.encode(text, allowed_special)
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Decoded text
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized. Train or load first.")
        
        return self.decoder.decode(token_ids)
    
    def decode_with_metadata(self, token_ids: List[int]) -> dict:
        """
        Decode token IDs with metadata.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            Dictionary with decoded text and token metadata
        """
        if not self._initialized:
            raise RuntimeError("Tokenizer not initialized. Train or load first.")
        
        return self.decoder.decode_with_metadata(token_ids)
    
    # ==================== Utility Methods ====================
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocab)
    
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert token string to ID."""
        return self.inverse_vocab.get(token)
    
    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert token ID to string."""
        return self.vocab.get(token_id)
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)