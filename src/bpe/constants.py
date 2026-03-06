"""Constants and default configurations for the BPE tokenizer."""
import os
# Special tokens
DEFAULT_SPECIAL_TOKENS = {"<|endoftext|>", "<unk>", "<pad>", "<s>", "</s>"}

# GPT-2 specific
GPT2_SPACE_TOKEN = "Ġ"  # Represents space in GPT-2 tokenization
NEWLINE_TOKEN = "\n"

CYRILLIC_UZBEK_RANGE = r'а-яА-ЯўҒғҚқҲҳЎ'
LATIN_UZBEK_RANGE = r'a-zA-ZāĀâÂçÇĞğĢģīĪİıŁłÑñŅņŌōŖŗŞşŠšŢţŪūŪŪŽž'
# Default vocabulary size for training
DEFAULT_VOCAB_SIZE = 50000

# Pre-tokenization patterns
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"""

# Url for downloading TinyStories dataset
TINY_STORIES_URL_TRAIN = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
TINY_STORIES_URL_VALID = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# WikiText-103 dataset URL
WIKITEXT103_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

# FineWeb dataset URL
FINEWEB_URL = "https://huggingface.co/datasets/DeepSeek/FineWeb/resolve/main/fineweb.txt"

PROCESSED_DATA_DIR = "data/processed"
PRO_TINY_STORIES_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
PRO_TINY_STORIES_VALID_PATH = os.path.join(PROCESSED_DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

TRAIN_DATA_DIR = "data/raw"
TINY_STORIES_TRAIN_PATH = os.path.join(TRAIN_DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
TINY_STORIES_VALID_PATH = os.path.join(TRAIN_DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
WIKITEXT103_PATH = os.path.join(TRAIN_DATA_DIR, "wikitext-103-raw-v1.zip")
FINEWEB_PATH = os.path.join(TRAIN_DATA_DIR, "fineweb.txt")