"""
vocab.py — Vocabulary management for GPT-4 BPE tokenizer.
Handles loading, saving, and querying the token <-> ID mappings.
"""

import json
from functools import lru_cache
import logging


class Vocab:
    def __init__(self):

        logging.info("Initializing Vocab instance...")
        self.vocab: dict[int, str] = {}          # id -> token string
        self.inverse_vocab: dict[str, int] = {}  # token string -> id

    def build_from_chars(self, chars: list[str]):
        """Initialize vocab from a list of characters (used during training)."""

        logging.info(f"Building vocab from {len(chars)} unique characters...")
        self.vocab = {i: ch for i, ch in enumerate(chars)}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}

    def add_token(self, token: str) -> int:
        """Add a new token and return its assigned ID."""
        if token in self.inverse_vocab:
            return self.inverse_vocab[token]
        new_id = len(self.vocab)
        self.vocab[new_id] = token
        self.inverse_vocab[token] = new_id
        return new_id

    def get_id(self, token: str) -> int | None:
        return self.inverse_vocab.get(token)

    def get_token(self, token_id: int) -> str | None:
        return self.vocab.get(token_id)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, token: str):
        return token in self.inverse_vocab

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token: str) -> int | None:
        return self.inverse_vocab.get(token)

    def save(self, path: str):
        logging.info(f"Saving vocab to {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        logging.info(f"Loading vocab from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data.items()}
        self.inverse_vocab = {v: int(k) for k, v in data.items()}

    def load_from_openai(self, path: str):
        """Load from OpenAI's encoder.json format (token -> id)."""
        logging.info(f"Loading OpenAI vocab from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {int(v): k for k, v in data.items()}
        self.inverse_vocab = {k: int(v) for k, v in data.items()}

        # GPT-4 uses cl100k_base which handles newlines natively,
        # but add fallback if missing
        if "\n" not in self.inverse_vocab:
            fallback = next(
                (t for t in ["<|endoftext|>", "Ġ", ""] if t in self.inverse_vocab),
                None
            )
            if fallback is None:
                raise KeyError("No fallback token found for '\\n'.")
            nid = self.inverse_vocab[fallback]
            self.inverse_vocab["\n"] = nid
            self.vocab[nid] = "\n"