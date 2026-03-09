"""
tokenizer.py — GPT-4 BPE Tokenizer.
Ties together Vocab, BPETrainer, Encoder, and Decoder.
"""

from .vocab import Vocab
from .train import BPETrainer
from .encode_decode import Encoder, Decoder
from typing import Iterator, Iterable
import logging


class GPT4Tokenizer:
    """
    GPT-4 style BPE tokenizer.

    Supports:
    - Training from scratch on raw text
    - Loading OpenAI's pretrained cl100k_base vocab + merges
    - Encoding text to token IDs
    - Decoding token IDs to text
    - Saving and loading custom trained tokenizers
    """

    def __init__(self):
        self.vocab = Vocab()
        self.bpe_merges: dict[tuple[int, int], int] = {}  # custom training merges
        self.bpe_ranks: dict[tuple[str, str], int] = {}   # OpenAI pretrained ranks

        self._trainer: BPETrainer | None = None
        self._encoder: Encoder | None = None
        self._decoder: Decoder | None = None

    def _build_components(self):

        logging.info("Building encoder and decoder components...")
        self._encoder = Encoder(self.vocab, self.bpe_merges, self.bpe_ranks)
        self._decoder = Decoder(self.vocab)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def train(
        self,
        text_iter: Iterator[str] | Iterable[str],
        vocab_size: int,
        allowed_special: set[str] = {""},
        chunk_size: int = 10_000
    ):
        """
        Train BPE tokenizer from scratch.
        
        Args:
            text_iter: An iterator that yields text chunks.
                  For large datasets, use an iterator to avoid loading all data into memory.
            vocab_size: Target vocabulary size.
            allowed_special: Special tokens to register in vocab.
            chunk_size: Number of lines to process per chunk when converting string to iterator.
        """

        logging.info("Training BPE tokenizer...")
        self._trainer = BPETrainer(self.vocab)
        self.bpe_merges = self._trainer.train(text_iter, vocab_size, allowed_special)
        self._build_components()

    # ------------------------------------------------------------------ #
    #  Load pretrained OpenAI weights
    # ------------------------------------------------------------------ #

    def load_from_openai(self, vocab_path: str, merges_path: str):
        """
        Load OpenAI's pretrained vocab and merges.

        Args:
            vocab_path: Path to encoder.json
            merges_path: Path to vocab.bpe
        """

        logging.info("Loading OpenAI pretrained vocab and merges...")
        self.vocab.load_from_openai(vocab_path)
        self._trainer = BPETrainer(self.vocab)
        self.bpe_ranks = self._trainer.load_merges_from_openai(merges_path)
        self._build_components()

    # ------------------------------------------------------------------ #
    #  Encode / Decode
    # ------------------------------------------------------------------ #

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """Encode text to token IDs."""
        if self._encoder is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load_from_openai() first.")
        return self._encoder.encode(text, allowed_special)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self._decoder is None:
            raise RuntimeError("Tokenizer not initialized. Call train() or load_from_openai() first.")
        return self._decoder.decode(token_ids)

    # ------------------------------------------------------------------ #
    #  Save / Load custom tokenizer
    # ------------------------------------------------------------------ #

    def save(self, vocab_path: str, merges_path: str):
        """Save vocab and merges to disk."""
        logging.info("Saving tokenizer...")
        self.vocab.save(vocab_path)
        if self._trainer:
            self._trainer.save_merges(merges_path)

    def load(self, vocab_path: str, merges_path: str):
        """Load a previously saved custom tokenizer."""

        logging.info(f"Loading custom tokenizer from {vocab_path} and {merges_path}...")
        self.vocab.load(vocab_path)
        self._trainer = BPETrainer(self.vocab)
        self._trainer.load_merges(merges_path)
        self.bpe_merges = self._trainer.bpe_merges
        self._build_components()

    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #

    def get_special_token_id(self, token: str) -> int | None:
        logging.info(f"Getting ID for special token: {token}")
        return self.vocab.get_special_token_id(token)

    def vocab_size(self) -> int:
        logging.info(f"Retrieving vocabulary size: {len(self.vocab)}")
        return len(self.vocab)