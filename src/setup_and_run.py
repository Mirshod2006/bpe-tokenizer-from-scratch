#!/usr/bin/env python3
"""
Setup and Training Pipeline for BPE Tokenizer

This script handles:
1. Setting up essential resources (directories, downloading datasets)
2. Preprocessing corpus data
3. Training BPE tokenizer
4. Saving vocabulary and merge pairs

Usage:
    python src/setup_and_run.py --setup              # Only setup (download data)
    python src/setup_and_run.py --train              # Only train (assumes data exists)
    python src/setup_and_run.py --setup --train      # Full pipeline (setup + train)
    python src/setup_and_run.py                      # Default: Full pipeline
"""
import logging
import os
import sys
import json
import argparse
from typing import Optional
from tqdm import tqdm
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe.tokenizer import GPT4Tokenizer
from bpe.constants import (
    TINY_STORIES_TRAIN_PATH,
    TRAIN_DATA_DIR,
    PROCESSED_DATA_DIR,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_SPECIAL_TOKENS
)
from bpe.download_data import fetch_tiny_stories_dataset
from bpe.utils import preprocess_corpus, logging_setup, read_corpus_in_chunks

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
VOCAB_FILE = os.path.join(MODELS_DIR, "vocab.json")
MERGES_FILE = os.path.join(MODELS_DIR, "merges.json")
TOKENIZER_CONFIG_FILE = os.path.join(MODELS_DIR, "tokenizer_config.json")
PRO_TINY_STORIES_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "tiny_stories_train_processed.txt")


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
    logging.info("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"✓ Created/verified directory: {directory}")

def setup():
    """
    Main setup function: Create directories and download datasets.
    
    Returns:
        bool: True if setup successful, False otherwise
    """
    logging.info("\n" + "="*80)
    logging.info("SETUP: PREPARING ESSENTIAL RESOURCES")
    logging.info("="*80 + "\n")
    
    try:
        # Step 1: Create directories
        logging.info("Step 1/2: Creating directories...")
        create_directories()
        
        # Step 2: Download datasets
        logging.info("\nStep 2/2: Downloading datasets...")
        fetch_tiny_stories_dataset()
        
        logging.info("\n" + "="*80)
        logging.info("✅ SETUP COMPLETE")
        logging.info("="*80 + "\n")
        return True
        
    except Exception as e:
        logging.error(f"\n✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_tokenizer(
    corpus_path: str,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    special_tokens: Optional[set[str]] = None,
    chunk_size: int = 10_000,
) -> GPT4Tokenizer:
    """
    Train BPE tokenizer on preprocessed corpus.
    
    Args:
        corpus_path: Path to preprocessed corpus file
        vocab_size: Target vocabulary size
        special_tokens: Set of special tokens to include
        chunk_size: Size of text chunks to process at a time
    Returns:
        Trained GPT4Tokenizer
    """

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    
    logging.info(f"\n🔧 Training BPE tokenizer...")
    logging.info(f"   Corpus: {corpus_path}")
    logging.info(f"   Target vocab size: {vocab_size:,}")

    # Set default special tokens if not provided
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS
    logging.info(f"   Special tokens: {special_tokens}")

    # Load corpus
    chunks = read_corpus_in_chunks(corpus_path, chunk_size=chunk_size)
    
    # Initialize tokenizer
    tokenizer = GPT4Tokenizer()
    # Train
    logging.info("\n   Training in progress...")
    tokenizer.train(
        text_iter=tqdm(chunks, desc="  Reading corpus", unit=" chunks"),
        vocab_size=vocab_size,
        allowed_special=special_tokens,
        chunk_size=chunk_size
    )
    
    actual_vocab_size = tokenizer.vocab_size()
    logging.info(f"   ✓ Training complete!")
    logging.info(f"   Actual vocab size: {actual_vocab_size:,}")
    
    return tokenizer


def save_tokenizer(tokenizer: GPT4Tokenizer, vocab_path: str, merges_path: str):
    """
    Save trained tokenizer to disk.
    
    Args:
        tokenizer: Trained GPT4Tokenizer
        vocab_path: Path to save vocabulary
        merges_path: Path to save merges
    """
    logging.info(f"\n💾 Saving tokenizer...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    
    # Save vocab and merges
    tokenizer.save(vocab_path, merges_path)
    
    logging.info(f"   ✓ Vocabulary saved: {vocab_path}")
    logging.info(f"   ✓ Merges saved: {merges_path}")
    
    # Save configuration
    config: dict[str,int | list[str] | str] = {
        "vocab_size": tokenizer.vocab_size(),
        "vocab_file": os.path.basename(vocab_path),
        "merges_file": os.path.basename(merges_path),
        "special_tokens": list(DEFAULT_SPECIAL_TOKENS),
        "model_type": "BPE"
    }
    
    with open(TOKENIZER_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logging.info(f"   ✓ Config saved: {TOKENIZER_CONFIG_FILE}")


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
    logging.info("\n" + "="*80)
    logging.info("TRAINING PIPELINE: PREPROCESS + TRAIN + SAVE")
    logging.info("="*80 + "\n")
    
    try:
        # Step 1: Preprocess corpus (if needed)
        if preprocess:
            logging.info("Step 1/3: Preprocessing corpus...")
            
            # Check if raw data exists
            if not os.path.exists(TINY_STORIES_TRAIN_PATH):
                logging.error(f"✗ Training data not found: {TINY_STORIES_TRAIN_PATH}")
                logging.info("   Run setup first: python setup&run.py --setup")
                return False
            
            preprocessed_path = preprocess_corpus(
                input_path=TINY_STORIES_TRAIN_PATH,
                output_path=PRO_TINY_STORIES_TRAIN_PATH,
                max_size_mb=max_corpus_size_mb
            )
        else:
            logging.info("Step 1/3: Using existing preprocessed corpus...")
            preprocessed_path = PRO_TINY_STORIES_TRAIN_PATH
            
            if not os.path.exists(preprocessed_path):
                logging.error(f"✗ Preprocessed data not found: {preprocessed_path}")
                logging.info("   Run with --preprocess flag")
                return False
        
        # Step 2: Train tokenizer
        logging.info("\nStep 2/3: Training tokenizer...")
        tokenizer = train_tokenizer(
            corpus_path=preprocessed_path,
            vocab_size=vocab_size,
            special_tokens=DEFAULT_SPECIAL_TOKENS
        )
        
        # Step 3: Save tokenizer
        logging.info("\nStep 3/3: Saving tokenizer...")
        save_tokenizer(
            tokenizer=tokenizer,
            vocab_path=VOCAB_FILE,
            merges_path=MERGES_FILE
        )
        
        # Test tokenizer
        logging.info("\n" + "="*80)
        logging.info("TESTING TOKENIZER")
        logging.info("="*80 + "\n")
        
        test_texts = [
            "Hello, world! This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "BPE tokenization is awesome!"
        ]
        
        for test_text in test_texts:
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            logging.info(f"Original: {test_text}")
            logging.info(f"Encoded:  {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
            logging.info(f"Decoded:  {decoded}")
            logging.info(f"Match:    {'✓' if decoded == test_text else '✗'}")
        
        logging.info("="*80)
        logging.info("✅ TRAINING PIPELINE COMPLETE")
        logging.info("="*80 + "\n")
        
        logging.info(f"📁 Model files saved to: {MODELS_DIR}/")
        logging.info(f"   - {os.path.basename(VOCAB_FILE)}")
        logging.info(f"   - {os.path.basename(MERGES_FILE)}")
        logging.info(f"   - {os.path.basename(TOKENIZER_CONFIG_FILE)}")
        
        return True
        
    except Exception as e:
        logging.error(f"\n✗ Training pipeline failed: {e}")
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
  python src/setup_and_run.py                          # Full pipeline (setup + train)
  python src/setup_and_run.py --setup                  # Only download data
  python src/setup_and_run.py --train                  # Only train (assumes data exists)
  python src/setup_and_run.py --vocab-size 30000       # Train with custom vocab size
  python src/setup_and_run.py --max-size 100           # Train on first 100MB only
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
    logging.info("\n" + "="*80)
    logging.info("BPE TOKENIZER - SETUP & TRAINING")
    logging.info("="*80)
    
    success = True
    
    # Run setup if requested
    if args.setup:
        success = setup()
        if not success:
            logging.error("\n❌ Setup failed. Exiting.")
            sys.exit(1)
    
    # Run training if requested
    if args.train:
        success = run_training_pipeline(
            vocab_size=args.vocab_size,
            preprocess=not args.no_preprocess,
            max_corpus_size_mb=args.max_size
        )
        if not success:
            logging.error("\n❌ Training failed. Exiting.")
            sys.exit(1)
    
    # Final message
    if success:
        logging.info("\n" + "="*80)
        logging.info("🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!")
        logging.info("="*80 + "\n")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    logging_setup()
    main()
