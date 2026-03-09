"""
train.py — BPE training logic for GPT-4 style tokenizer.
Trains a vocabulary from raw text using byte-level BPE.
"""

from .vocab import Vocab
from .utils import preprocess_text_gpt4, bytes_to_unicode
from collections import defaultdict
from typing import Iterator, Iterable
from tqdm import tqdm
import logging


class BPETrainer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.bpe_merges: dict[tuple[int, int], int] = {}

    def train(
        self,
        text_iter: Iterator[str] | Iterable[str],
        vocab_size: int,
        allowed_special: set[str] = {"<|endoftext|>"}
    ):
        """
        Train BPE from scratch on streaming text input.

        Args:
            text_iter: Iterator that yields text chunks.
            vocab_size: Target vocabulary size.
            allowed_special: Special tokens to register in vocab.
        """

        logging.info("Starting BPE training...")
        # Step 1: Byte-level base vocab (256 bytes -> unicode chars)
        byte_encoder: dict[int, str] = bytes_to_unicode()  # {int -> str}
        unique_chars = list(byte_encoder.values())  # 256 base tokens

        self.vocab.build_from_chars(unique_chars)

        # Step 2: Register special tokens
        for token in (allowed_special or []):
            self.vocab.add_token(token)

        # Step 3 & 4: Build word frequency dict from streaming input
        # Instead of storing full corpus as list of lists, deduplicate words using dict
        logging.info("Preprocessing text and building word frequency dict...")
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        
        for chunk in text_iter:
            words = preprocess_text_gpt4(chunk)
            for word in words:
                # Encode word to byte-level token IDs
                byte_chars = [byte_encoder[b] for b in word.encode("utf-8")]
                ids = tuple(
                    token_id for ch in byte_chars 
                    if (token_id := self.vocab.get_id(ch)) is not None
                )
                if ids:
                    word_freqs[ids] += 1

        logging.info(f"Built word frequency dict with {len(word_freqs)} unique words")

        # Step 5: Build initial pair_counts ONCE before merge loop
        logging.info("Building initial pair counts...")
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += freq

        # Step 6: BPE merge loop with INCREMENTAL updates
        logging.info("Performing BPE merges...")
        merges_done = 0
        with tqdm(total=vocab_size - len(self.vocab), desc="BPE merges", unit=" merges") as pbar:
            while len(self.vocab) < vocab_size:
                if not pair_counts:
                    break
                
                # Get most frequent pair
                pair = max(pair_counts, key=pair_counts.__getitem__)
                if pair_counts[pair] <= 0:
                    break
                
                token1 = self.vocab.get_token(pair[0])
                token2 = self.vocab.get_token(pair[1])
                if token1 is None or token2 is None:
                    break
                
                new_id = self.vocab.add_token(token1 + token2)
                self.bpe_merges[pair] = new_id

                # Incrementally update word_freqs and pair_counts
                # Only process words that contain the merged pair
                new_word_freqs: dict[tuple[int, ...], int] = {}
                for word, freq in word_freqs.items():
                    new_word: list[int] = []
                    i = 0
                    while i < len(word):
                        # Check if current position matches the pair to merge
                        if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                            # Update pair counts for neighbors of the merge
                            if i > 0:
                                # Remove old left-neighbor pair
                                pair_counts[(word[i-1], word[i])] -= freq
                                # Add new left-neighbor pair
                                pair_counts[(word[i-1], new_id)] += freq
                            if i < len(word) - 2:
                                # Remove old right-neighbor pair
                                pair_counts[(word[i+1], word[i+2])] -= freq
                                # Add new right-neighbor pair
                                pair_counts[(new_id, word[i+2])] += freq

                            new_word.append(new_id)
                            i += 2  # Skip both tokens in the pair
                        else:
                            new_word.append(word[i])
                            i += 1

                    new_word_freqs[tuple(new_word)] = freq
                    new_word.clear()  # Free memory

                # Mark merged pair as consumed
                pair_counts[pair] = 0
                word_freqs = new_word_freqs
                new_word_freqs.clear()  # Free memory

                merges_done += 1
                if merges_done % 500 == 0:
                    logging.info(f"  Completed {merges_done} merges, vocab size: {len(self.vocab)}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "vocab": len(self.vocab),
                    "top_pair_freq": pair_counts.get(pair, 0),
                })

        logging.info(f"BPE training completed. Final vocab size: {len(self.vocab)}")
        return self.bpe_merges

    def save_merges(self, path: str):
        import json
        merges_list: list[dict[str, list[int] | int]] = [{"pair": list(p), "new_id": nid}
                       for p, nid in self.bpe_merges.items()]
        logging.info(f"Saving merges to {path}")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)

    def load_merges(self, path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} merges from {path}")
        self.bpe_merges = {tuple(m["pair"]): m["new_id"] for m in data}

    def load_merges_from_openai(self, bpe_merges_path: str) -> dict[tuple[str, str], int]:
        """
        Load GPT-2/GPT-4 style merge file and return bpe_ranks.
        bpe_ranks: {(str_token1, str_token2): rank}
        """

        logging.info(f"Loading BPE merges from OpenAI file: {bpe_merges_path}")
        bpe_ranks: dict[tuple[str, str], int] = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines and lines[0].startswith("#"):
            lines = lines[1:]

        logging.info(f"Processing {len(lines)} merge pairs from OpenAI merges file...")
        for rank, line in enumerate(lines):
            pair = tuple(line.strip().split())
            if len(pair) == 2:
                t1, t2 = pair
                if t1 in self.vocab and t2 in self.vocab:
                    bpe_ranks[(t1, t2)] = rank
        logging.info(f"Loaded {len(bpe_ranks)} valid merges from OpenAI file.")
        return bpe_ranks