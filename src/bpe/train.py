"""
train.py — BPE training logic for GPT-4 style tokenizer.
Trains a vocabulary from raw text using byte-level BPE.
"""

from .vocab import Vocab
from .utils import find_freq_pair, replace_pair, preprocess_text_gpt4, bytes_to_unicode


class BPETrainer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.bpe_merges: dict[tuple[int, int], int] = {}

    def train(self, text: str, vocab_size: int, allowed_special: set[str] = {"<|endoftext|>"}):
        """
        Train BPE from scratch on raw text.

        Args:
            text: Raw training corpus.
            vocab_size: Target vocabulary size.
            allowed_special: Special tokens to register in vocab.
        """
        # Step 1: Byte-level base vocab (256 bytes -> unicode chars)
        byte_encoder = bytes_to_unicode()  # {int -> str}
        unique_chars = list(byte_encoder.values())  # 256 base tokens

        self.vocab.build_from_chars(unique_chars)

        # Step 2: Register special tokens
        for token in (allowed_special or []):
            self.vocab.add_token(token)

        # Step 3: Pre-tokenize text with GPT-4 regex pattern
        words = preprocess_text_gpt4(text)

        # Step 4: Encode each word to byte-level token IDs
        def word_to_ids(word: str) -> list[int]:
            byte_chars = [byte_encoder[b] for b in word.encode("utf-8")]
            return [token_id for ch in byte_chars if (token_id := self.vocab.get_id(ch)) is not None]

        # Build corpus as list of token ID sequences (one per word)
        corpus = [word_to_ids(w) for w in words]

        # Step 5: BPE merge loop
        while len(self.vocab) < vocab_size:
            # Flatten corpus to find global most frequent pair
            all_ids = [tid for seq in corpus for tid in seq]
            pair = find_freq_pair(all_ids, mode="most")
            if pair is None:
                break

            token1 = self.vocab.get_token(pair[0])
            token2 = self.vocab.get_token(pair[1])
            if token1 is not None and token2 is not None:
                new_id = self.vocab.add_token(token1 + token2)
            else:
                break
            self.bpe_merges[pair] = new_id

            # Apply merge to entire corpus
            corpus = [replace_pair(seq, pair, new_id) for seq in corpus]

        return self.bpe_merges

    def save_merges(self, path: str):
        import json
        merges_list: list[dict[str, list[int] | int]] = [{"pair": list(p), "new_id": nid}
                       for p, nid in self.bpe_merges.items()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

    def load_merges(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.bpe_merges = {tuple(m["pair"]): m["new_id"] for m in data}

    def load_merges_from_openai(self, bpe_merges_path: str) -> dict[tuple[str, str], int]:
        """
        Load GPT-2/GPT-4 style merge file and return bpe_ranks.
        bpe_ranks: {(str_token1, str_token2): rank}
        """
        bpe_ranks: dict[tuple[str, str], int] = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines and lines[0].startswith("#"):
            lines = lines[1:]

        for rank, line in enumerate(lines):
            pair = tuple(line.strip().split())
            if len(pair) == 2:
                t1, t2 = pair
                if t1 in self.vocab and t2 in self.vocab:
                    bpe_ranks[(t1, t2)] = rank
        return bpe_ranks