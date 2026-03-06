#!/usr/bin/env python3
"""
Demonstration script for text preprocessing filters.
Shows the effect of each filter step-by-step.
"""

import sys
sys.path.insert(0, '/home/mirshod/Desktop/bpe-tokenizer-from-scratch')

from scripts.preprocess_corpus import (
    normalize_unicode,
    decode_html_entities,
    normalize_urls_and_emails,
    remove_control_characters,
    normalize_whitespace,
    pretokenize_with_gpt2_pattern,
    pretokenize_simple,
    preprocess_text_string
)


def demonstrate_filters():
    """Show each filter's effect on sample text."""
    
    # Sample text with various issues
    sample_text = """
    Hello!   This is a test    with     multiple spaces.
    
    
    Check out this URL: https://example.com/page?id=123
    Email me at: test@example.com
    
    HTML entities: &lt;div&gt; &amp; &quot;quotes&quot; &#39;apostrophe&#39;
    
    Contractions: I'm, you're, we've, can't, won't, they'll
    
    Unicode variants: café vs café (different representations)
    
    Numbers: 123 456 789.01
    
    	Tabs	and		mixed		whitespace
    
    Special chars: @#$%^&*()
    """
    
    print("=" * 80)
    print("ORIGINAL TEXT:")
    print("=" * 80)
    print(repr(sample_text))
    print()
    
    # Step 1: Unicode normalization
    print("=" * 80)
    print("STEP 1: Unicode Normalization (NFC)")
    print("=" * 80)
    text = normalize_unicode(sample_text, form='NFC')
    print(repr(text))
    print()
    
    # Step 2: HTML entity decoding
    print("=" * 80)
    print("STEP 2: HTML Entity Decoding")
    print("=" * 80)
    text = decode_html_entities(text)
    print(repr(text))
    print()
    
    # Step 3: URL/Email normalization
    print("=" * 80)
    print("STEP 3: URL and Email Normalization")
    print("=" * 80)
    text = normalize_urls_and_emails(text, replace_with=' <URL> ')
    print(repr(text))
    print()
    
    # Step 4: Control character removal
    print("=" * 80)
    print("STEP 4: Control Character Removal")
    print("=" * 80)
    text = remove_control_characters(text)
    print(repr(text))
    print()
    
    # Step 5: Whitespace normalization
    print("=" * 80)
    print("STEP 5: Whitespace Normalization")
    print("=" * 80)
    text = normalize_whitespace(text)
    print(repr(text))
    print()
    
    # Step 6: Pre-tokenization
    print("=" * 80)
    print("STEP 6: Pre-tokenization with GPT2 Pattern")
    print("=" * 80)
    try:
        tokens = pretokenize_with_gpt2_pattern(text)
        print(f"Number of tokens: {len(tokens)}")
        print("First 30 tokens:")
        for i, token in enumerate(tokens[:30], 1):
            print(f"  {i:2d}. {repr(token)}")
    except Exception as e:
        print(f"Error with GPT2 pattern: {e}")
        print("\nFalling back to simple tokenization:")
        tokens = pretokenize_simple(text)
        print(f"Number of tokens: {len(tokens)}")
        print("First 30 tokens:")
        for i, token in enumerate(tokens[:30], 1):
            print(f"  {i:2d}. {repr(token)}")
    
    print()
    print("=" * 80)
    print("FINAL RESULT (cleaned text string):")
    print("=" * 80)
    final_text = preprocess_text_string(sample_text)
    print(final_text)
    print()


def test_uzbek_text():
    """Test with Uzbek text (both Latin and Cyrillic)."""
    
    uzbek_samples = [
        "Salom dunyo! Men O'zbekistondan keldim.",  # Latin
        "Салом дунё! Мен Ўзбекистондан келдим.",  # Cyrillic
        "Visit https://uzbekistan.uz for more info. Email: info@uzbekistan.uz"
    ]
    
    print("=" * 80)
    print("UZBEK TEXT EXAMPLES")
    print("=" * 80)
    
    for i, text in enumerate(uzbek_samples, 1):
        print(f"\nSample {i}:")
        print(f"Original: {text}")
        
        cleaned = preprocess_text_string(text)
        print(f"Cleaned:  {cleaned}")
        
        try:
            tokens = pretokenize_with_gpt2_pattern(cleaned)
            print(f"Tokens:   {tokens[:15]}")  # First 15 tokens
        except Exception:
            tokens = pretokenize_simple(cleaned)
            print(f"Tokens (simple): {tokens[:15]}")


if __name__ == "__main__":
    print("\n🔧 TEXT PREPROCESSING FILTERS DEMONSTRATION\n")
    
    demonstrate_filters()
    print("\n" + "=" * 80 + "\n")
    test_uzbek_text()
    
    print("\n✅ Filter demonstration complete!\n")
