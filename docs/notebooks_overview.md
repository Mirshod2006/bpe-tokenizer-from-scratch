# Notebooks Overview

Interactive Jupyter notebooks for learning BPE and experimenting with the tokenizer.

## Notebooks

### 1. `bpe_algorithm_explanation.ipynb`

**Purpose**: Step-by-step introduction to the BPE algorithm.

**Topics**:
- Why tokenization matters for LLMs
- Pre-tokenization with GPT-style regex
- Iterative pair merging
- Building vocabulary from merges
- Visualizations (merge trees, frequency charts)

**Use when**: Learning BPE from scratch or teaching others.

---

### 2. `karpathy_tokenizer_exp.ipynb`

**Purpose**: Karpathy-style exploration of tokenization quirks.

**Topics**:
- Why LLMs struggle with spelling, reversing strings, arithmetic
- Unicode and byte-level encoding
- Raw bytes → integer IDs
- Pair frequency and merge operations
- Simple encoder/decoder from scratch
- GPT-2 regex patterns
- Comparing tiktoken (GPT-2, cl100k_base)
- SentencePiece training and output

**Use when**: Understanding tokenization edge cases and real-world behavior.

---

### 3. `tokenization_examples.ipynb`

**Purpose**: Practical examples using the project's tokenizer.

**Topics**:
- TinyStories sample texts
- Encoding and decoding
- Comparison with tiktoken and Hugging Face tokenizers
- Edge cases and behavior analysis
- Visualization of token boundaries

**Use when**: Testing the `GPT4Tokenizer` and comparing with other tokenizers.

---

### 4. `training_visualization.ipynb`

**Purpose**: Visualize the BPE training process.

**Topics**:
- Loading TinyStories data
- Training on small samples
- Vocabulary evolution over merges
- Frequency distribution of tokens
- Merge statistics and charts

**Use when**: Understanding how the vocabulary is built during training.

---

## Running Notebooks

From the project root:

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to `notebooks/` and open any `.ipynb` file.

**Note**: Some notebooks import from `src.bpe` (e.g. `GPT4Tokenizer`). Run from the project root or ensure it is in `sys.path`. The main tokenizer class is `GPT4Tokenizer` (not `BPETokenizer`).

## Dependencies

Notebooks may use additional packages:
- `matplotlib`, `seaborn` — plotting
- `pandas` — data tables
- `networkx` — graph visualizations (BPE algorithm)
- `tiktoken` — comparison
- `transformers` — Hugging Face (optional)
- `datasets` — Hugging Face datasets (optional)

Install as needed:
```bash
pip install matplotlib seaborn pandas networkx tiktoken
```
