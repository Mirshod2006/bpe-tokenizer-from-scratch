#!/usr/bin/env python3
"""
Setup and Training Pipeline for BPE Tokenizer

This script handles:
1. Setting up essential resources (directories, downloading datasets)
2. Preprocessing corpus data
3. Training BPE tokenizer
4. Saving vocabulary and merge pairs

Usage:
    python setup&run.py --setup              # Only setup (download data)
    python setup&run.py --train              # Only train (assumes data exists)
    python setup&run.py --setup --train      # Full pipeline (setup + train)
    python setup&run.py                      # Default: Full pipeline
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.bpe.tokenizer import BPETokenizer
from src.bpe.constants import (
    TINY_STORIES_URL_TRAIN,
    TINY_STORIES_URL_VALID,
    TINY_STORIES_TRAIN_PATH,
    TINY_STORIES_VALID_PATH,
    PRO_TINY_STORIES_TRAIN_PATH,
    PRO_TINY_STORIES_VALID_PATH,
    TRAIN_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_SPECIAL_TOKENS
)
from scripts.preprocess_corpus import preprocess_corpus
from scripts.download_data import download_file, fetch_tiny_stories_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "src/models"
VOCAB_FILE = os.path.join(MODELS_DIR, "vocab.json")
MERGES_FILE = os.path.join(MODELS_DIR, "merges.json")
TOKENIZER_CONFIG_FILE = os.path.join(MODELS_DIR, "tokenizer_config.json")


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def create_directories():
    """Create all necessary directories for data and models."""
    directories = [
        TRAIN_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        "data",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created/verified directory: {directory}")

def setup():
    """
    Main setup function: Create directories and download datasets.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    print("\n" + "="*80)
    print("SETUP: PREPARING ESSENTIAL RESOURCES")
    print("="*80 + "\n")
    
    try:
        # Step 1: Create directories
        print("Step 1/2: Creating directories...")
        create_directories()
        
        # Step 2: Download datasets
        print("\nStep 2/2: Downloading datasets...")
        fetch_tiny_stories_dataset()
        
        print("\n" + "="*80)
        print("✅ SETUP COMPLETE")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_tokenizer(
    corpus_path: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: Optional[set] = None
) -> BPETokenizer:
    """
    Train BPE tokenizer on preprocessed corpus.
    
    Args:
        corpus_path: Path to preprocessed corpus file
        vocab_size: Target vocabulary size
        special_tokens: Set of special tokens to include
    
    Returns:
        Trained BPETokenizer
    """
    print(f"\n🔧 Training BPE tokenizer...")
    print(f"   Corpus: {corpus_path}")
    print(f"   Target vocab size: {vocab_size:,}")
    
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   Corpus length: {len(text):,} characters")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer()
    
    # Set default special tokens if not provided
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS
    
    print(f"   Special tokens: {special_tokens}")
    
    # Train
    print("\n   Training in progress...")
    tokenizer.train(
        text=text,
        vocab_size=vocab_size,
        allowed_special=special_tokens
    )
    
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"   ✓ Training complete!")
    print(f"   Actual vocab size: {actual_vocab_size:,}")
    
    return tokenizer


def save_tokenizer(tokenizer: BPETokenizer, vocab_path: str, merges_path: str):
    """
    Save trained tokenizer to disk.
    
    Args:
        tokenizer: Trained BPETokenizer
        vocab_path: Path to save vocabulary
        merges_path: Path to save merges
    """
    print(f"\n💾 Saving tokenizer...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    # Save vocab and merges
    tokenizer.save(vocab_path, merges_path)
    
    print(f"   ✓ Vocabulary saved: {vocab_path}")
    print(f"   ✓ Merges saved: {merges_path}")
    
    # Save configuration
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "vocab_file": os.path.basename(vocab_path),
        "merges_file": os.path.basename(merges_path),
        "special_tokens": list(DEFAULT_SPECIAL_TOKENS),
        "model_type": "BPE"
    }
    
    with open(TOKENIZER_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Config saved: {TOKENIZER_CONFIG_FILE}")


def run_training_pipeline(
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    preprocess: bool = True,
    max_corpus_size_mb: Optional[int] = None
):
    """
    Main training pipeline: preprocess corpus, train tokenizer, save model.
    
    Args:
        vocab_size: Target vocabulary size
        preprocess: Whether to preprocess raw data (or use existing processed data)
        max_corpus_size_mb: Optional limit on corpus size for faster training
    
    Returns:
        bool: True if training successful, False otherwise
    """
    print("\n" + "="*80)
    print("TRAINING PIPELINE: PREPROCESS + TRAIN + SAVE")
    print("="*80 + "\n")
    
    try:
        # Step 1: Preprocess corpus (if needed)
        # if preprocess:
        #     print("Step 1/3: Preprocessing corpus...")
            
        #     # Check if raw data exists
        #     if not os.path.exists(TINY_STORIES_TRAIN_PATH):
        #         print(f"✗ Training data not found: {TINY_STORIES_TRAIN_PATH}")
        #         print("   Run setup first: python setup&run.py --setup")
        #         return False
            
        #     preprocessed_path = preprocess_corpus(
        #         input_path=TINY_STORIES_TRAIN_PATH,
        #         output_path=PRO_TINY_STORIES_TRAIN_PATH,
        #         max_size_mb=max_corpus_size_mb
        #     )
        # else:
        #     print("Step 1/3: Using existing preprocessed corpus...")
        #     preprocessed_path = PRO_TINY_STORIES_TRAIN_PATH
            
        #     if not os.path.exists(preprocessed_path):
        #         print(f"✗ Preprocessed data not found: {preprocessed_path}")
        #         print("   Run with --preprocess flag")
        #         return False
        
        # Step 2: Train tokenizer
        print("\nStep 2/3: Training tokenizer...")
        tokenizer = train_tokenizer(
            corpus_path=TINY_STORIES_TRAIN_PATH,
            vocab_size=vocab_size,
            special_tokens=DEFAULT_SPECIAL_TOKENS
        )
        
        # Step 3: Save tokenizer
        print("\nStep 3/3: Saving tokenizer...")
        save_tokenizer(
            tokenizer=tokenizer,
            vocab_path=VOCAB_FILE,
            merges_path=MERGES_FILE
        )
        
        # Test tokenizer
        print("\n" + "="*80)
        print("TESTING TOKENIZER")
        print("="*80 + "\n")
        
        test_texts = [
            "Hello, world! This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "BPE tokenization is awesome!"
        ]
        
        for test_text in test_texts:
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            print(f"Original: {test_text}")
            print(f"Encoded:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
            print(f"Decoded:  {decoded}")
            print(f"Match:    {'✓' if decoded == test_text else '✗'}")
            print()
        
        print("="*80)
        print("✅ TRAINING PIPELINE COMPLETE")
        print("="*80 + "\n")
        
        print(f"📁 Model files saved to: {MODELS_DIR}/")
        print(f"   - {os.path.basename(VOCAB_FILE)}")
        print(f"   - {os.path.basename(MERGES_FILE)}")
        print(f"   - {os.path.basename(TOKENIZER_CONFIG_FILE)}")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Setup and train BPE tokenizer on TinyStories dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup&run.py                          # Full pipeline (setup + train)
  python setup&run.py --setup                  # Only download data
  python setup&run.py --train                  # Only train (assumes data exists)
  python setup&run.py --vocab-size 30000       # Train with custom vocab size
  python setup&run.py --max-size 100           # Train on first 100MB only
        """
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run setup (create directories and download datasets)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training pipeline (preprocess, train, save)'
    )
    
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f'Target vocabulary size (default: {DEFAULT_VOCAB_SIZE})'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=None,
        help='Maximum corpus size in MB for faster training (default: no limit)'
    )
    
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Skip preprocessing and use existing processed data'
    )
    
    args = parser.parse_args()
    
    # If no flags specified, run both setup and train
    if not args.setup and not args.train:
        args.setup = True
        args.train = True
    
    # Header
    print("\n" + "="*80)
    print("BPE TOKENIZER - SETUP & TRAINING")
    print("="*80)
    
    success = True
    
    # Run setup if requested
    if args.setup:
        success = setup()
        if not success:
            print("\n❌ Setup failed. Exiting.")
            sys.exit(1)
    
    # Run training if requested
    if args.train:
        success = run_training_pipeline(
            vocab_size=args.vocab_size,
            preprocess=not args.no_preprocess,
            max_corpus_size_mb=args.max_size
        )
        if not success:
            print("\n❌ Training failed. Exiting.")
            sys.exit(1)
    
    # Final message
    if success:
        print("\n" + "="*80)
        print("🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
