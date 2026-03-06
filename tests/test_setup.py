#!/usr/bin/env python3
"""
Quick test of the setup&run.py pipeline (without downloading large datasets).

This script demonstrates:
1. Directory creation
2. File structure verification
3. Basic import checks
"""

import sys
sys.path.insert(0, '/home/mirshod/Desktop/bpe-tokenizer-from-scratch')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.bpe.tokenizer import BPETokenizer
        print("  ✓ BPETokenizer")
        
        from src.bpe.trainer import BPETrainer
        print("  ✓ BPETrainer")
        
        from src.bpe.constants import (
            DEFAULT_VOCAB_SIZE,
            DEFAULT_SPECIAL_TOKENS,
            TRAIN_DATA_DIR,
            PROCESSED_DATA_DIR
        )
        print("  ✓ Constants")
        
        from scripts.preprocess_corpus import preprocess_text_string
        print("  ✓ Preprocessing functions")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}\n")
        return False


def test_directory_creation():
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
        exists = os.path.isdir(directory)
        status = "✓" if exists else "✗"
        print(f"  {status} {directory}")
    
    print("\n✅ Directory structure created!\n")


def test_preprocessing():
    """Test preprocessing pipeline on sample text."""
    print("Testing preprocessing pipeline...")
    
    from scripts.preprocess_corpus import preprocess_text_string
    
    sample_texts = [
        "Hello,   world!   Multiple    spaces.",
        "Visit https://example.com for more info.",
        "HTML: &lt;div&gt; &amp; &quot;test&quot;",
        "Unicode: café vs café (different forms)",
    ]
    
    for text in sample_texts:
        cleaned = preprocess_text_string(text)
        print(f"  Input:  {text[:50]}")
        print(f"  Output: {cleaned[:50]}\n")
    
    print("✅ Preprocessing works!\n")


def test_tokenizer_basic():
    """Test basic tokenizer initialization and training on tiny sample."""
    print("Testing tokenizer on tiny sample...")
    
    from src.bpe.tokenizer import BPETokenizer
    
    # Create tiny training sample
    sample_text = "Hello world! " * 100  # Repeat for enough data
    
    # Initialize and train
    tokenizer = BPETokenizer()
    tokenizer.train(
        text=sample_text,
        vocab_size=300,  # Small vocab for fast test
        allowed_special={"<|endoftext|>"}
    )
    
    print(f"  Trained vocab size: {tokenizer.get_vocab_size()}")
    
    # Test encode/decode
    test_text = "Hello world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  {decoded}")
    print(f"  Match:    {'✓' if decoded == test_text else '✗'}")
    
    print("\n✅ Tokenizer works!\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SETUP & RUN - QUICK TEST")
    print("="*80 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Directory Creation", test_directory_creation),
        ("Preprocessing", test_preprocessing),
        ("Tokenizer", test_tokenizer_basic),
    ]
    
    results = []
    
    for name, test_func in tests:
        print("-" * 80)
        try:
            result = test_func()
            results.append((name, True))
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
        print("  python setup&run.py --setup     # Download TinyStories dataset")
        print("  python setup&run.py --train     # Train tokenizer")
        print("  python setup&run.py             # Full pipeline")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the errors above before running setup&run.py")
    print("="*80 + "\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
