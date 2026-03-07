"""
encode_decode.py — Encode and decode logic for GPT-4 BPE tokenizer.
Handles text -> token IDs and token IDs -> text.
"""

import re
from .vocab import Vocab
from .utils import preprocess_text_gpt4, bytes_to_unicode
import logging

class Encoder:
    def __init__(self, vocab: Vocab, bpe_merges: dict[tuple[int, int], int], bpe_ranks: dict[tuple[str, str], int]):
        """
        Args:
            vocab: Shared Vocab instance.
            bpe_merges: {(id1, id2): merged_id} — used for custom-trained tokenizers.
            bpe_ranks: {(str1, str2): rank} — used for OpenAI pretrained tokenizers.
        """
        self.vocab: Vocab = vocab
        self.bpe_merges: dict[tuple[int, int], int] = bpe_merges
        self.bpe_ranks: dict[tuple[str, str], int] = bpe_ranks
        self._byte_encoder: dict[int, str] = bytes_to_unicode()  # byte int -> unicode char

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input string.
            allowed_special: Set of special tokens to handle explicitly.

        Returns:
            List of token IDs.
        """
        token_ids: list[int] = []

        logging.info(f"Encoding text: {text[:50]}{'...' if len(text) > 50 else ''}")
        # Handle special tokens first
        if allowed_special:
            special_pattern = (
                "(" + "|".join(re.escape(t) for t in sorted(allowed_special, key=len, reverse=True)) + ")"
            )
            parts = re.split(special_pattern, text)

            logging.info(f"Split text into parts based on allowed special tokens.")
            for part in parts:
                if part in allowed_special:
                    sid = self.vocab.get_id(part)
                    if sid is None:
                        raise ValueError(f"Special token '{part}' not in vocab.")
                    token_ids.append(sid)
                else:
                    token_ids.extend(self._encode_ordinary(part))
        else:
            token_ids.extend(self._encode_ordinary(text))

        return token_ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text that contains no special tokens."""
        token_ids: list[int] = []

        logging.info(f"Encoding ordinary text: {text[:20]}{'...' if len(text) > 20 else ''}")
        words = preprocess_text_gpt4(text)
        for word in words:
            word_ids = self._tokenize_word(word)
            token_ids.extend(word_ids)
        return token_ids

    def _tokenize_word(self, word: str) -> list[int]:
        """
        Convert a single pre-tokenized word to token IDs using BPE.
        GPT-4 encodes at byte level first, then applies BPE merges.
        """
        # Byte-encode the word, then map each byte to its unicode symbol
        byte_chars = [self._byte_encoder[b] for b in word.encode("utf-8")]

        # Look up each character in vocab
        token_ids: list[int] = []

        for ch in byte_chars:
            tid = self.vocab.get_id(ch)
            if tid is None:
                raise ValueError(f"Byte char '{ch}' not found in vocab.")
            token_ids.append(tid)

        # Apply BPE merges
        if self.bpe_ranks:
            return self._apply_bpe_ranked(byte_chars)
        else:
            return self._apply_bpe_merges(token_ids)

    def _apply_bpe_merges(self, token_ids: list[int]) -> list[int]:
        """Greedy BPE merge using bpe_merges dict (custom training)."""
        can_merge: bool = True
        while can_merge and len(token_ids) > 1:
            can_merge = False
            new_tokens: list[int] = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i + 1])
                if pair in self.bpe_merges:
                    new_tokens.append(self.bpe_merges[pair])
                    i += 2
                    can_merge = True
                else:
                    new_tokens.append(token_ids[i])
                    i += 1
            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens
        return token_ids

    def _apply_bpe_ranked(self, symbols: list[str]) -> list[int]:
        """
        BPE merge using bpe_ranks (OpenAI style).
        Always merges the lowest-rank (highest priority) pair first.
        """
        while len(symbols) > 1:
            pairs = set(zip(symbols, symbols[1:]))
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        result: list[int] = []
        for s in symbols:
            tid = self.vocab.get_id(s)
            if tid is None:
                raise ValueError(f"Symbol '{s}' not found in vocab.")
            result.append(tid)
        return result


class Decoder:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        # Reverse byte encoder: unicode char -> original byte
        byte_enc = bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in byte_enc.items()}

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to a UTF-8 string.
        GPT-4 decodes by reconstructing the byte sequence, then decoding UTF-8.
        """
        tokens: list[str] = []

        logging.info(f"Decoding token IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        for tid in token_ids:
            token = self.vocab.get_token(tid)
            if token is None:
                raise ValueError(f"Token ID {tid} not found in vocab.")
            tokens.append(token)

        # Concatenate all token strings, then map back from unicode symbols to bytes
        text = "".join(tokens)
        byte_values = [self._byte_decoder.get(ch, ord(ch)) for ch in text]
        
        logging.info(f"Decoded bytes: {byte_values[:10]}{'...' if len(byte_values) > 20 else ''}")
        return bytes(byte_values).decode("utf-8", errors="replace")