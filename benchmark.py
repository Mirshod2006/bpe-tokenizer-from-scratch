#!/usr/bin/env python3
"""
Benchmarking Script for BPE Tokenizer

This script compares the custom BPE tokenizer with various tiktoken tokenizers
on multiple metrics including speed, compression ratio, and token count.

Usage:
    python benchmark.py
    python benchmark.py --sample-size 1000
    python benchmark.py --output models/benchmark_results.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


import tiktoken
from tiktoken import Encoding

from src.bpe.tokenizer import GPT4Tokenizer


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
VOCAB_FILE = os.path.join(MODELS_DIR, "vocab.json")
MERGES_FILE = os.path.join(MODELS_DIR, "merges.json")
BENCHMARK_OUTPUT = os.path.join(MODELS_DIR, "benchmark_results.json")

# Sample texts for testing
SAMPLE_TEXTS = [
    "Once upon a time, in a small village, there lived a little girl named Lucy.",
    "The quick brown fox jumps over the lazy dog. This is a pangram sentence.",
    "Machine learning and artificial intelligence are transforming the world of technology.",
    "Hello, world! This is a test of the tokenizer with special characters: @#$%^&*()",
    "In the year 2024, scientists discovered a new planet in a distant galaxy.",
    "Python is a high-level, interpreted programming language known for its simplicity.",
    "She sells seashells by the seashore. The shells she sells are surely seashells.",
    "To be or not to be, that is the question posed by Shakespeare's Hamlet.",
    "The rain in Spain stays mainly in the plain, as the saying goes.",
    "A journey of a thousand miles begins with a single step, according to ancient wisdom."
]


# ============================================================================
# TIKTOKEN TOKENIZERS TO BENCHMARK
# ============================================================================

TIKTOKEN_MODELS = [
    "gpt2",           # GPT-2 tokenizer
    "r50k_base",      # GPT-3 tokenizer (Codex)
    "p50k_base",      # GPT-3 tokenizer (text models)
    "cl100k_base",    # GPT-3.5/GPT-4 tokenizer
    "o200k_base",     # GPT-4o tokenizer
]


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================

def load_test_data(data_path: str = '', sample_size: int = 500) -> str:
    """
    Load test data for benchmarking.
    
    Args:
        data_path: Path to text file. If None, uses default samples.
        sample_size: Number of characters to sample from the file.
    
    Returns:
        Test text string
    """
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read(sample_size * 100)  # Read more to get sample_size words
            words = text.split()[:sample_size]
            return ' '.join(words)
    else:
        # Use predefined samples
        return '\n'.join(SAMPLE_TEXTS * 10)  # Repeat for more data


def benchmark_tokenizer(tokenizer: GPT4Tokenizer | Encoding, tokenizer_name: str, test_texts: List[str]) -> Dict[str, Any]:
    """
    Benchmark a tokenizer on various metrics.
    
    Args:
        tokenizer: Tokenizer instance (custom or tiktoken)
        tokenizer_name: Name of the tokenizer
        test_texts: List of test texts
    
    Returns:
        Dictionary with benchmark results
    """
    results: dict[str, Any] = {
        "tokenizer_name": tokenizer_name,
        "total_texts": len(test_texts),
        "total_characters": sum(len(text) for text in test_texts),
        "total_tokens": 0,
        "avg_tokens_per_text": 0,
        "compression_ratio": 0,  # chars per token
        "tokenization_time_ms": 0,
        "tokens_per_second": 0,
        "per_text_results": []
    }
    
    all_token_counts: list[int] = []
    start_time = time.time()
    
    for text in test_texts:
        text_start = time.time()
        
        # Encode text
        try:
            tokens: list[int] = tokenizer.encode(text, allowed_special={""})
        except Exception as e:
            print(f"Error encoding with {tokenizer_name}: {e}")
            tokens: list[int] = []
        
        text_time = (time.time() - text_start) * 1000  # ms

        token_count = len(tokens)
        all_token_counts.append(token_count)
        
        results["per_text_results"].append({
            "text_length": len(text),
            "token_count": token_count,
            "compression_ratio": len(text) / token_count if token_count > 0 else 0,
            "time_ms": round(text_time, 4)
        })
    
    total_time = (time.time() - start_time) * 1000  # ms
    
    results["total_tokens"] = sum(all_token_counts)
    results["avg_tokens_per_text"] = round(sum(all_token_counts) / len(test_texts), 2)
    results["compression_ratio"] = round(results["total_characters"] / results["total_tokens"], 2) if results["total_tokens"] > 0 else 0
    results["tokenization_time_ms"] = round(total_time, 2)
    results["tokens_per_second"] = round(results["total_tokens"] / (total_time / 1000), 2) if total_time > 0 else 0
    
    return results


def load_custom_tokenizer(vocab_path: str, merges_path: str) -> GPT4Tokenizer:
    """Load the custom BPE tokenizer."""
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(
            f"Custom tokenizer files not found. Please train the tokenizer first.\n"
            f"Expected files:\n  - {vocab_path}\n  - {merges_path}"
        )
    
    tokenizer = GPT4Tokenizer()
    tokenizer.load(vocab_path, merges_path)
    print(f"✓ Loaded custom BPE tokenizer (vocab size: {tokenizer.vocab_size()})")
    return tokenizer


def run_benchmark(
    test_texts: List[str],
    vocab_path: str = VOCAB_FILE,
    merges_path: str = MERGES_FILE,
    include_tiktoken: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark comparing custom tokenizer with tiktoken models.
    
    Args:
        test_texts: List of test texts
        vocab_path: Path to custom vocab file
        merges_path: Path to custom merges file
        include_tiktoken: Whether to include tiktoken benchmarks
    
    Returns:
        Dictionary with all benchmark results
    """
    benchmark_results: dict[str, Any] = {
        "benchmark_info": {
            "total_test_texts": len(test_texts),
            "total_characters": sum(len(t) for t in test_texts),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tokenizers": {}
    }
    
    # Benchmark custom tokenizer
    print("\n" + "=" * 70)
    print("BENCHMARKING CUSTOM BPE TOKENIZER")
    print("=" * 70)
    
    try:
        custom_tokenizer: GPT4Tokenizer = load_custom_tokenizer(vocab_path, merges_path)
        custom_results = benchmark_tokenizer(custom_tokenizer, "Custom BPE", test_texts)
        benchmark_results["tokenizers"]["custom_bpe"] = custom_results
        
        print(f"\n📊 Custom BPE Results:")
        print(f"  Total Tokens: {custom_results['total_tokens']}")
        print(f"  Avg Tokens/Text: {custom_results['avg_tokens_per_text']}")
        print(f"  Compression Ratio: {custom_results['compression_ratio']:.2f} chars/token")
        print(f"  Speed: {custom_results['tokens_per_second']:.2f} tokens/sec")
        print(f"  Time: {custom_results['tokenization_time_ms']:.2f} ms")
    except Exception as e:
        print(f"❌ Error benchmarking custom tokenizer: {e}")
    
    # Benchmark tiktoken models
    if include_tiktoken:
        print("\n" + "=" * 70)
        print("BENCHMARKING TIKTOKEN TOKENIZERS")
        print("=" * 70)
        
        for model_name in TIKTOKEN_MODELS:
            try:
                print(f"\n🔍 Loading {model_name}...")
                tiktoken_enc: Encoding = tiktoken.get_encoding(model_name)  # type: Encoding
                results: Dict[str, Any] = benchmark_tokenizer(tiktoken_enc, model_name, test_texts)
                benchmark_results["tokenizers"][model_name] = results
                
                print(f"  Total Tokens: {results['total_tokens']}")
                print(f"  Avg Tokens/Text: {results['avg_tokens_per_text']}")
                print(f"  Compression Ratio: {results['compression_ratio']:.2f} chars/token")
                print(f"  Speed: {results['tokens_per_second']:.2f} tokens/sec")
            except Exception as e:
                print(f"  ❌ Error with {model_name}: {e}")
    
    return benchmark_results


def create_comparison_matrix(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a comparison matrix from benchmark results.
    
    Returns:
        Dictionary with comparative metrics
    """
    tokenizers = results.get("tokenizers", {})
    
    if not tokenizers:
        return {"error": "No tokenizer results found"}
    
    comparison: dict[str, Any] = {
        "summary": {},
        "rankings": {
            "best_compression": None,
            "fastest_speed": None,
            "lowest_token_count": None,
        },
        "detailed_comparison": {}
    }
    
    # Extract metrics
    for name, data in tokenizers.items():
        comparison["detailed_comparison"][name] = {
            "total_tokens": data["total_tokens"],
            "avg_tokens_per_text": data["avg_tokens_per_text"],
            "compression_ratio": data["compression_ratio"],
            "tokens_per_second": data["tokens_per_second"],
            "tokenization_time_ms": data["tokenization_time_ms"]
        }
    
    # Find best performers
    if tokenizers:
        # Best compression (highest chars/token ratio)
        best_compression = max(tokenizers.items(), key=lambda x: x[1]["compression_ratio"])
        comparison["rankings"]["best_compression"] = {
            "tokenizer": best_compression[0],
            "ratio": best_compression[1]["compression_ratio"]
        }
        
        # Fastest speed
        fastest = max(tokenizers.items(), key=lambda x: x[1]["tokens_per_second"])
        comparison["rankings"]["fastest_speed"] = {
            "tokenizer": fastest[0],
            "tokens_per_second": fastest[1]["tokens_per_second"]
        }
        
        # Lowest token count (most efficient)
        lowest_tokens = min(tokenizers.items(), key=lambda x: x[1]["total_tokens"])
        comparison["rankings"]["lowest_token_count"] = {
            "tokenizer": lowest_tokens[0],
            "total_tokens": lowest_tokens[1]["total_tokens"]
        }
    
    return comparison


def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add comparison matrix
    results["comparison_matrix"] = create_comparison_matrix(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print a summary of benchmark results."""
    comparison = results.get("comparison_matrix", {})
    rankings = comparison.get("rankings", {})
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    if rankings.get("best_compression"):
        print(f"\n🏆 Best Compression: {rankings['best_compression']['tokenizer']}")
        print(f"   Ratio: {rankings['best_compression']['ratio']:.2f} chars/token")
    
    if rankings.get("fastest_speed"):
        print(f"\n⚡ Fastest Speed: {rankings['fastest_speed']['tokenizer']}")
        print(f"   Speed: {rankings['fastest_speed']['tokens_per_second']:.2f} tokens/sec")
    
    if rankings.get("lowest_token_count"):
        print(f"\n📉 Most Efficient (Lowest Tokens): {rankings['lowest_token_count']['tokenizer']}")
        print(f"   Tokens: {rankings['lowest_token_count']['total_tokens']}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark BPE tokenizer against tiktoken models")
    parser.add_argument("--vocab", default=VOCAB_FILE, help="Path to custom vocab file")
    parser.add_argument("--merges", default=MERGES_FILE, help="Path to custom merges file")
    parser.add_argument("--output", default=BENCHMARK_OUTPUT, help="Output JSON file path")
    parser.add_argument("--data", help="Path to test data file")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of words to sample")
    parser.add_argument("--no-tiktoken", action="store_true", help="Skip tiktoken benchmarks")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BPE TOKENIZER BENCHMARK")
    print("=" * 70)
    
    # Load test data
    if args.data:
        print(f"\n📖 Loading test data from: {args.data}")
        test_text = load_test_data(args.data, args.sample_size)
        test_texts = [test_text]
    else:
        print(f"\n📖 Using default sample texts ({len(SAMPLE_TEXTS)} samples)")
        test_texts = SAMPLE_TEXTS
    
    # Run benchmarks
    results = run_benchmark(
        test_texts,
        vocab_path=args.vocab,
        merges_path=args.merges,
        include_tiktoken=not args.no_tiktoken
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output)
    
    print("\n✨ Benchmark complete!")


if __name__ == "__main__":
    main()
