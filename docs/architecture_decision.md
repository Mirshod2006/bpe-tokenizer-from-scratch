# Architecture Decision: Single File vs. Split Files

## Current Implementation: Single File (✅ Recommended)

The `setup&run.py` file contains both setup and training functionality in **one file**.

### Structure

```python
setup&run.py
├── Configuration (constants, paths)
├── Setup Functions
│   ├── create_directories()
│   ├── download_file()
│   ├── fetch_tiny_stories_dataset()
│   └── setup()
├── Preprocessing Functions
│   └── preprocess_corpus()
├── Training Functions
│   ├── train_tokenizer()
│   ├── save_tokenizer()
│   └── run_training_pipeline()
└── CLI Interface
    └── main()
```

## Why Single File is Recommended

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
   - One command for everything: `python setup&run.py`
   - Clear interface with flags
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

- Larger file (~500 lines)
- Less modular (but still well-organized)
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

### Example Split Implementation

**config.py**
```python
"""Shared configuration for setup and training."""
MODELS_DIR = "src/models"
VOCAB_SIZE = 50000
# ... other constants
```

**setup.py**
```python
"""Data download and directory setup."""
from config import *

def setup():
    create_directories()
    fetch_datasets()
    # ...
```

**train.py**
```python
"""Tokenizer training pipeline."""
from config import *
from setup import create_directories

def train():
    preprocess_corpus()
    train_tokenizer()
    save_model()
    # ...
```

**main.py**
```python
"""CLI interface."""
import setup
import train

def main():
    if args.setup:
        setup.setup()
    if args.train:
        train.train()
```

## Current Recommendation: Keep Single File

For this project, **keep the single file** because:

1. **Project Size**: ~500 lines is manageable
2. **Clear Purpose**: Pipeline has one job
3. **User Experience**: Simple `setup&run.py` command
4. **Maintenance**: Easier to maintain cohesive flow

## Migration Path (If Needed Later)

If you decide to split later:

```bash
# 1. Create new structure
mkdir -p scripts/pipeline

# 2. Move functions to separate files
# - setup.py: setup functions
# - train.py: training functions
# - config.py: constants

# 3. Create main.py that imports and orchestrates

# 4. Keep setup&run.py as simple wrapper:
from scripts.pipeline import setup, train, main
if __name__ == "__main__":
    main()
```

## Code Organization Best Practices

Current file uses:

### ✅ Clear Sections
```python
# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

# ============================================================================
# PREPROCESSING FUNCTIONS  
# ============================================================================

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
```

### ✅ Focused Functions
- Each function does one thing
- Clear docstrings
- Type hints
- Error handling

### ✅ Logical Flow
1. Imports
2. Configuration
3. Setup functions
4. Preprocessing functions
5. Training functions
6. CLI interface

### ✅ Reusable Components
- Functions can be imported: `from setup&run import setup, train_tokenizer`
- Not tied to CLI
- Testable independently

## Conclusion

**Keep it simple**: One file works best for this use case.

The current implementation balances:
- ✅ Simplicity for users
- ✅ Maintainability for developers
- ✅ Extensibility for future growth
- ✅ Clarity of purpose

**When to reconsider**: If file grows beyond 1000 lines or complexity increases significantly.
