# Architecture Decision: Single File vs. Split Files

## Current Implementation: Single Pipeline File (✅ Recommended)

The `src/setup_and_run.py` file contains both setup and training functionality in **one file**.

### Structure

```python
setup_and_run.py
├── Configuration (MODELS_DIR, VOCAB_FILE, paths)
├── Setup Functions
│   ├── create_directories()
│   ├── setup() → create_directories() + fetch_tiny_stories_dataset()
│   └── (uses bpe.download_data)
├── Training Functions
│   ├── train_tokenizer()     # Uses GPT4Tokenizer, read_corpus_in_chunks
│   ├── save_tokenizer()
│   └── run_training_pipeline()  # Preprocess, train, save
└── CLI Interface
    └── main()
```

### Source Module Layout

```
src/bpe/
├── tokenizer.py      # GPT4Tokenizer (orchestrator)
├── train.py          # BPETrainer (BPE merge logic)
├── encode_decode.py  # Encoder, Decoder
├── vocab.py          # Vocab (token ↔ ID)
├── utils.py          # preprocess_text_gpt4, read_corpus_in_chunks, etc.
├── constants.py      # Paths, URLs, special tokens
└── download_data.py  # fetch_tiny_stories_dataset, etc.
```

## Why Single Pipeline File is Recommended

### ✅ Advantages

1. **Sequential Operations**
   - Setup → Preprocess → Train → Save
   - Natural pipeline flow
   - Easier to understand execution order

2. **Shared Configuration**
   - All paths defined in one place
   - No duplication of constants
   - Consistent settings across phases

3. **Single Entry Point**
   - One command: `python src/setup_and_run.py`
   - Clear interface with `--setup`, `--train`, `--vocab-size`, etc.
   - Less confusion for users

4. **Simpler Dependencies**
   - No circular imports
   - Clear function hierarchy
   - Easy to trace execution

5. **Development Convenience**
   - Modify pipeline in one place
   - See entire workflow at a glance
   - Easier debugging

### ⚠️ Limitations

- Larger file (~370 lines)
- Less modular (but BPE logic is in separate modules)
- Can't import setup separately without CLI

## Alternative: Split Files

If the project grows significantly, consider splitting:

### Proposed Structure

```
scripts/
├── setup.py          # Setup and data download
├── train.py          # Training pipeline
├── config.py         # Shared configuration
└── utils.py          # Helper functions
```

### When to Split

Split into multiple files if:
- File exceeds 1000 lines
- Multiple people editing simultaneously
- Need to import setup without training
- Building a larger framework
- Reusing components in different contexts

## Current Recommendation: Keep Single File

For this project, **keep the single pipeline file** because:

1. **Project Size**: ~370 lines is manageable
2. **Clear Purpose**: Pipeline has one job
3. **User Experience**: Simple `python src/setup_and_run.py` command
4. **Maintenance**: BPE logic is already modular in `src/bpe/`

## Code Organization Best Practices

### ✅ Clear Sections

```python
# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
```

### ✅ Focused Functions

- Each function does one thing
- Clear docstrings
- Type hints
- Error handling

### ✅ Reusable Components

- `src/bpe/` modules can be imported independently
- `train_tokenizer()`, `save_tokenizer()` are testable
- Not tied to CLI

## Conclusion

**Keep it simple**: One pipeline file works best for this use case.

The current implementation balances:
- ✅ Simplicity for users
- ✅ Maintainability (BPE in separate modules)
- ✅ Extensibility for future growth
- ✅ Clarity of purpose

**When to reconsider**: If `setup_and_run.py` grows beyond 1000 lines or complexity increases significantly.
