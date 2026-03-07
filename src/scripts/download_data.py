import os
import sys
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from bpe.constants import (TINY_STORIES_URL_TRAIN, TINY_STORIES_URL_VALID, WIKITEXT103_URL, 
                               FINEWEB_URL,
                               TINY_STORIES_TRAIN_PATH, TINY_STORIES_VALID_PATH, 
                               WIKITEXT103_PATH, FINEWEB_PATH)


# ==================== Downloading Datasets ====================

def download_file(url: str, output_path: str, chunk_size: int = 8192):
    """
    Download a file from URL with progress indication.
    
    Args:
        url: URL to download from
        output_path: Local path to save file
        chunk_size: Size of chunks to download
    """
    print(f"📥 Downloading: {url}")
    print(f"   Saving to: {output_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size if available
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
        
        print(f"\n✓ Downloaded successfully: {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error downloading {url}: {e}")
        raise


def fetch_tiny_stories_dataset():
    """Download TinyStories training and validation datasets."""
    print("\n" + "="*80)
    print("DOWNLOADING TINYSTORIES DATASET")
    print("="*80 + "\n")
    
    # Check if files already exist
    if os.path.exists(TINY_STORIES_TRAIN_PATH) and os.path.exists(TINY_STORIES_VALID_PATH):
        print("⚠ Dataset files already exist. Skipping download.")
        print(f"  Train: {TINY_STORIES_TRAIN_PATH}")
        print(f"  Valid: {TINY_STORIES_VALID_PATH}")
        
        response = input("\nOverwrite existing files? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Download training set
    download_file(TINY_STORIES_URL_TRAIN, TINY_STORIES_TRAIN_PATH)
    
    # Download validation set
    download_file(TINY_STORIES_URL_VALID, TINY_STORIES_VALID_PATH)
    
    print("\n✅ TinyStories dataset downloaded successfully!")

def fetch_wikitext103_dataset():
    """Download WikiText-103 dataset."""
    print("\n" + "="*80)
    print("DOWNLOADING WIKITEXT-103 DATASET")
    print("="*80 + "\n")
    
    # Check if files already exist
    if os.path.exists(WIKITEXT103_PATH):
        print("⚠ Dataset file already exists. Skipping download.")
        print(f"  File: {WIKITEXT103_PATH}")
        
        response = input("\nOverwrite existing file? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Download training set
    download_file(WIKITEXT103_URL, WIKITEXT103_PATH)
    
    print("\n✅ WikiText-103 dataset downloaded successfully!")

def fetch_fineweb_dataset():
    """Download FineWeb dataset."""
    print("\n" + "="*80)
    print("DOWNLOADING FINEWEB DATASET")
    print("="*80 + "\n")
    
    # Check if files already exist
    if os.path.exists(FINEWEB_PATH):
        print("⚠ Dataset file already exists. Skipping download.")
        print(f"  File: {FINEWEB_PATH}")
        
        response = input("\nOverwrite existing file? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
    
    # Download training set
    download_file(FINEWEB_URL, FINEWEB_PATH)