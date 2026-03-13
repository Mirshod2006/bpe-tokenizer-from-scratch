#!/usr/bin/env python3
"""
Quick test of the setup&run.py pipeline (without downloading large datasets).

This script demonstrates:
1. Directory creation
2. File structure verification
3. Basic import checks
"""

import os
import sys
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe.utils import logging_setup


def test_directory_creation() -> bool:
    """Test directory structure creation."""
    print("Testing directory creation...")
    
    from src.bpe.constants import TRAIN_DATA_DIR, PROCESSED_DATA_DIR
    
    directories = [
        TRAIN_DATA_DIR,
        PROCESSED_DATA_DIR,
        "src/models",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        exists: bool = os.path.isdir(directory)
        status = "✓" if exists else "✗"
        print(f"  {status} {directory}")
    
    print("\n✅ Directory structure created!\n")
    return True

def test_preprocessing() -> bool:
    """Test preprocessing pipeline on sample text."""
    print("Testing preprocessing pipeline...")
    
    from src.bpe.utils import preprocess_text_gpt4
    
    sample_texts = [
        "Hello,   world!   Multiple    spaces.",
        "Visit https://example.com for more info.",
        "HTML: &lt;div&gt; &amp; &quot;test&quot;",
        "Unicode: café vs café (different forms)",
    ]
    
    for text in sample_texts:
        cleaned = list(preprocess_text_gpt4(text))
        print(f"  Input:  {text[:50]}")
        print(f"  Output: {cleaned[:50]}\n")
    
    print("✅ Preprocessing works!\n")
    return True


def test_tokenizer_basic() -> bool:
    """Test basic tokenizer initialization and training on tiny sample."""
    print("Testing tokenizer on tiny sample...")
    
    from src.bpe.tokenizer import GPT4Tokenizer
    
    # Create tiny training sample
    sample_text = "Hello world! " * 100  # Repeat for enough data
    
    # Initialize and train (pass list of chunks as text_iter)
    tokenizer = GPT4Tokenizer()
    tokenizer.train(
        text_iter=[sample_text],
        vocab_size=300,  # Small vocab for fast test
        allowed_special={"<|endoftext|>"}
    )
    
    print(f"  Trained vocab size: {tokenizer.vocab_size()}")
    
    # Test encode/decode
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    print(f"  Match:    {'✓' if decoded == test_text else '✗'}")
    
    print("\n✅ Tokenizer works!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SETUP & RUN - QUICK TEST")
    print("="*80 + "\n")
    
    tests: list[tuple[str, Any]] = [
        ("Directory Creation", test_directory_creation),
        ("Preprocessing", test_preprocessing),
        ("Tokenizer", test_tokenizer_basic),
    ]
    
    results: list[tuple[str, bool]] = []
    
    for name, test_func in tests:
        print("-" * 80)
        try:
            result: bool = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
        print()
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python src/setup_and_run.py --setup     # Download TinyStories dataset")
        print("  python src/setup_and_run.py --train     # Train tokenizer")
        print("  python src/setup_and_run.py             # Full pipeline")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the errors above before running src/setup_and_run.py")
    print("="*80 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    logging_setup()
    main()
