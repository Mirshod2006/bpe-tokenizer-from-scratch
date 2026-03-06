import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bpe.constants import (TINY_STORIES_URL_TRAIN, TINY_STORIES_URL_VALID, WIKITEXT103_URL, 
                               FINEWEB_URL,
                               TINY_STORIES_TRAIN_PATH, TINY_STORIES_VALID_PATH, 
                               WIKITEXT103_PATH, FINEWEB_PATH)


# ==================== Downloading Datasets ====================

def download_file(url: str, output_path: str):
    """Download a file from a URL to a specified output path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded: {output_path}")

def fetch_tiny_stories():
    """Download TinyStories dataset."""
    os.makedirs("data", exist_ok=True)
    download_file(TINY_STORIES_URL_TRAIN, TINY_STORIES_TRAIN_PATH)
    download_file(TINY_STORIES_URL_VALID, TINY_STORIES_VALID_PATH)

def fetch_wikitext103():
    """Download WikiText-103 dataset."""
    os.makedirs("data", exist_ok=True)
    download_file(WIKITEXT103_URL, WIKITEXT103_PATH)

def fetch_fineweb():
    """Download FineWeb dataset."""
    os.makedirs("data", exist_ok=True)
    download_file(FINEWEB_URL, FINEWEB_PATH)