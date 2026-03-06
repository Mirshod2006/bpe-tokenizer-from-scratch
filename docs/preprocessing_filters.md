# Text Preprocessing Filters

Complete guide to the preprocessing pipeline implemented in `scripts/preprocess_corpus.py`.

## Filter Pipeline Order

The filters are applied in a specific order to ensure optimal text cleaning:

```
Raw Text
    ↓
1. Unicode Normalization
    ↓
2. HTML Entity Decoding
    ↓
3. URL/Email Normalization
    ↓
4. Control Character Removal
    ↓
5. Whitespace Normalization
    ↓
6. Pre-tokenization (Regex Pattern)
    ↓
Clean Tokens
```

---

## 1. Unicode Normalization

**Function:** `normalize_unicode(text, form='NFC')`

**Purpose:** Standardizes different Unicode representations of the same characters.

**Forms:**
- `NFC` (Canonical Composition) - Recommended for most cases
- `NFKC` (Compatibility Composition) - More aggressive normalization

**Example:**
```python
# Different representations of "café"
text1 = "café"  # é as single character (U+00E9)
text2 = "café"  # é as e + combining acute (U+0065 U+0301)
# After NFC normalization, both become identical
```

**Why First?** Ensures all subsequent operations work with standardized text.

---

## 2. HTML Entity Decoding

**Function:** `decode_html_entities(text)`

**Purpose:** Converts HTML entities to their actual characters.

**Examples:**
- `&lt;` → `<`
- `&gt;` → `>`
- `&amp;` → `&`
- `&quot;` → `"`
- `&#39;` → `'`

**Why Second?** After Unicode normalization, so decoded characters are also normalized.

---

## 3. URL and Email Normalization

**Function:** `normalize_urls_and_emails(text, replace_with=' <URL> ')`

**Purpose:** 
- Reduces vocabulary size
- Prevents learning domain-specific patterns
- Maintains semantic meaning

**Patterns Detected:**
- URLs: `http://`, `https://`, `ftp://`, `www.`
- Emails: `user@domain.com`

**Examples:**
```python
"Visit https://example.com" → "Visit <URL>"
"Email test@example.com" → "Email <EMAIL>"
```

**Configuration:**
- Set `replace_with=None` to keep original URLs/emails
- Customize replacement token as needed

**Why Third?** After HTML decoding to handle encoded URLs properly.

---

## 4. Control Character Removal

**Function:** `remove_control_characters(text, keep_chars={'\n', '\t', '\r'})`

**Purpose:** Removes non-printable characters while preserving important whitespace.

**Preserved by Default:**
- `\n` - Newlines (paragraph structure)
- `\t` - Tabs (later converted to spaces)
- `\r` - Carriage returns

**Removed:**
- Null bytes
- ASCII control codes
- Zero-width characters
- Other non-printable Unicode

**Why Fourth?** After content transformations but before whitespace normalization.

---

## 5. Whitespace Normalization

**Function:** `normalize_whitespace(text)`

**Purpose:** Standardizes all whitespace for consistent tokenization.

**Transformations:**
1. Tabs → Single spaces
2. Multiple spaces → Single space
3. `\r\n` and `\r` → `\n`
4. 3+ newlines → Max 2 newlines (paragraph breaks)
5. Spaces at line boundaries removed

**Examples:**
```python
"Hello    world" → "Hello world"
"Line1\n\n\n\nLine2" → "Line1\n\nLine2"
"Text\t\twith\ttabs" → "Text with tabs"
```

**Why Fifth?** Last step before tokenization to ensure clean boundaries.

---

## 6. Pre-tokenization (Regex Pattern)

**Function:** `pretokenize_with_gpt2_pattern(text)`

**Purpose:** Split text into tokens while preventing merging across important boundaries.

**GPT2 Pattern Handles:**
- Contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
- Words: Optional space + letters
- Numbers: Optional space + digits
- Punctuation: Optional space + symbols
- Whitespace sequences

**Boundaries Preserved:**
- Word boundaries
- Whitespace boundaries
- Punctuation boundaries
- Number grouping

**Examples:**
```python
"I'm happy!" → ["I", "'m", " happy", "!"]
"Number 123.45" → [" Number", " 123", ".", "45"]
"Hello world" → ["Hello", " world"]
```

**Fallback:** If `regex` module unavailable, uses simplified pattern.

**Why Last?** All text cleaning must be complete before splitting into tokens.

---

## Usage Examples

### Basic Usage

```python
from scripts.preprocess_corpus import preprocess_corpus

# Process a corpus file
tokens = preprocess_corpus('data/raw/corpus.txt')
```

### Custom Configuration

```python
from scripts.preprocess_corpus import preprocess_corpus

# Keep URLs, use NFKC normalization, use simple tokenization
tokens = preprocess_corpus(
    'data/raw/corpus.txt',
    normalize_urls=False,      # Keep original URLs
    unicode_form='NFKC',       # Aggressive normalization
    use_gpt2_pattern=False     # Simple tokenization
)
```

### Process String (Without Tokenization)

```python
from scripts.preprocess_corpus import preprocess_text_string

text = "Hello   &lt;world&gt;!   Visit https://example.com"
clean = preprocess_text_string(text)
# Result: "Hello <world>! Visit <URL>"
```

### Individual Filters

```python
from scripts.preprocess_corpus import (
    normalize_unicode,
    decode_html_entities,
    normalize_whitespace
)

text = "café &amp; café"
text = normalize_unicode(text)
text = decode_html_entities(text)
# Result: "café & café"
```

---

## Testing

Run the demonstration script to see all filters in action:

```bash
python scripts/test_filters.py
```

This will show:
- Step-by-step transformation of sample text
- Effect of each filter
- Pre-tokenization with GPT2 pattern
- Examples with Uzbek text (Latin and Cyrillic)

---

## Dependencies

- **Python 3.6+**
- **regex** module (for GPT2 pattern with Unicode properties)
  ```bash
  pip install regex
  ```
- **unicodedata** (built-in)
- **html** (built-in)

---

## Performance Considerations

1. **Order Matters:** Filters are ordered for optimal efficiency
2. **Unicode Form:** NFC is faster than NFKC
3. **Regex Module:** ~2-3x faster than pure Python patterns
4. **Batch Processing:** Process large files in chunks if memory-constrained

---

## Language Support

The filters support all Unicode languages including:

- **Latin scripts:** English, French, Spanish, German, etc.
- **Cyrillic scripts:** Russian, Ukrainian, Uzbek Cyrillic, etc.
- **Other scripts:** Arabic, Chinese, Japanese, Korean, etc.

**Uzbek-specific:** Both Latin (O'zbek) and Cyrillic (Ўзбек) are fully supported.

---

## Best Practices

1. **Always apply filters in order** - Don't skip steps
2. **Test on sample data first** - Verify output matches expectations
3. **Preserve special tokens** - Add to vocabulary before training
4. **Document custom patterns** - If you modify patterns, document why
5. **Version control** - Track changes to preprocessing logic

---

## Troubleshooting

### `regex` module not found
```bash
pip install regex
# Or for system Python:
python3 -m pip install --break-system-packages regex
```

### Memory issues with large files
Process in chunks:
```python
def process_large_file(file_path, chunk_size=10000):
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield preprocess_text_string(chunk)
```

### Slow processing
- Use NFC instead of NFKC
- Disable URL normalization if not needed
- Use simple pattern instead of GPT2 pattern

---

## References

- [GPT-2 Tokenization](https://github.com/openai/gpt-2)
- [Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- [Python regex module](https://pypi.org/project/regex/)
