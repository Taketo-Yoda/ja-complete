# ja-complete

Lightweight offline Japanese input completion library without LLM or database.

## Overview

`ja-complete` is a pure Python OSS library for Japanese text completion/prediction. It provides multiple independent completion methods for different use cases:

- **Phrase-based completion**: Custom phrase lists with automatic prefix generation
- **N-gram model**: Statistical predictions based on Japanese text corpus
- **Simple dictionary**: Direct prefix-to-suggestion mapping
- **Custom JSONL**: Flexible custom data format support

Key features:
- No LLM or database required: Fully offline and lightweight
- Multiple independent completion APIs
- Morphological analysis powered by Janome
- Easy integration into CLI tools, editors, or web applications

## Installation

```bash
pip install ja-complete
# or with uv
uv add ja-complete
```

## Quick Start

### Phrase-based Completion

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# Add custom phrases
phrases = [
    "スマホの買い換えと合わせて一式揃えたい",
    "新生活に備えた準備を始めたい",
    "夏を爽やかに過ごしたい",
]
completer.add_phrases(phrases)

# Get completions (automatically falls back to N-gram if no phrase matches)
results = completer.suggest_from_phrases("ス", top_k=5)
print(results)  # [{'text': 'スマホの買い換えと合わせて一式揃えたい', 'score': 0.82}, ...]

# Disable N-gram fallback for strict phrase matching
results = completer.suggest_from_phrases("未登録の入力", fallback_to_ngram=False)
print(results)  # [] (empty if no phrase matches)
```

### N-gram Completion

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# Use default N-gram model
results = completer.suggest_from_ngram("今日は", top_k=5)
print(results)  # [{'text': '今日はいい天気', 'score': 0.85}, ...]
```

### Simple Dictionary Completion

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# Add simple prefix mappings
suggestions = {
    "お": ["おはよう", "おやすみ", "お疲れ様"],
    "あり": ["ありがとう", "ありがとうございます"],
}
completer.add_simple_suggestions(suggestions)

# Get completions (automatically falls back to N-gram if no match)
results = completer.suggest_from_simple("あり", top_k=3)
print(results)  # [{'text': 'ありがとう', 'score': 1.0}, ...]

# Disable fallback
results = completer.suggest_from_simple("未登録", fallback_to_ngram=False)
print(results)  # [] (empty if no match)
```

### Converting Phrases to JSONL

```python
from ja_complete import JaCompleter

# Convert phrases to JSONL format for N-gram model training
phrases = [
    "今日はいい天気ですね",
    "明日は雨が降りそうです",
    "週末は晴れるといいな",
]

jsonl = JaCompleter.convert_to_jsonl(phrases)
print(jsonl)
# Output:
# {"text": "今日はいい天気ですね", "tokens": ["今日", "は", "いい", "天気", "です", "ね"]}
# {"text": "明日は雨が降りそうです", "tokens": ["明日", "は", "雨", "が", "降り", "そう", "です"]}
# {"text": "週末は晴れるといいな", "tokens": ["週末", "は", "晴れる", "と", "いい", "な"]}

# Save to file for model training
with open("phrases.jsonl", "w", encoding="utf-8") as f:
    f.write(jsonl)
```

## CLI Usage

```bash
# Phrase-based completion
ja-complete phrase "新生活" --phrases phrases.txt

# N-gram completion
ja-complete ngram "今日は"

# Simple dictionary completion
ja-complete simple "あり" --dict suggestions.json
```

## API Reference

### JaCompleter

Main class providing multiple completion methods.

#### Constructor

- `JaCompleter(enable_ngram_fallback: bool = True)`
  - Initialize completer with optional N-gram fallback
  - When `enable_ngram_fallback=True`, phrase and simple dictionary methods automatically use N-gram completions when no matches are found

#### Methods

**Phrase-based Completion:**

- `add_phrases(phrases: List[str]) -> None`
  - Add phrases for phrase-based completion
  - Automatically generates prefixes using morphological analysis

- `suggest_from_phrases(input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None) -> List[Dict[str, Any]]`
  - Get completions from added phrases
  - If no matches and `fallback_to_ngram=True` (or instance default), returns N-gram completions
  - Returns ranked results with scores

**N-gram Completion:**

- `suggest_from_ngram(input_text: str, top_k: int = 10) -> List[Dict[str, Any]]`
  - Get completions using N-gram model
  - Uses default model or custom model if loaded

- `load_ngram_model(model_path: str) -> None`
  - Load custom N-gram model from file

**Simple Dictionary Completion:**

- `add_simple_suggestions(suggestions: Dict[str, List[str]]) -> None`
  - Add prefix-to-suggestions mapping

- `suggest_from_simple(input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None) -> List[Dict[str, Any]]`
  - Get completions from simple dictionary
  - If no matches and `fallback_to_ngram=True` (or instance default), returns N-gram completions
  - Direct prefix matching

**Utility Methods:**

- `convert_to_jsonl(phrases: List[str]) -> str` (static method)
  - Convert list of phrases to JSONL format
  - Each line contains: `{"text": "phrase", "tokens": ["token1", "token2", ...]}`
  - Useful for preparing training data for N-gram models

## How Scoring Works

### Phrase-based Completion Scoring

The phrase-based completion uses a hybrid scoring algorithm that considers both prefix matching and semantic similarity:

**Score Components:**
1. **Prefix Match Quality (60%)**: How well the input matches the beginning of the phrase
2. **Morpheme Overlap (40%)**: How many morphemes (word units) from your input appear in the phrase

**Examples:**

```python
# Long input with perfect morpheme overlap
Input: "スマホの買い換え"
Phrase: "スマホの買い換えと合わせて一式揃えたい"
Score: 0.82 (high - good prefix match + all morphemes present)

# Short input
Input: "スマホ"
Phrase: "スマホの買い換えと合わせて一式揃えたい"
Score: 0.75 (good - shorter prefix but morpheme is present)

# Perfect match
Input: "夏を爽やかに過ごしたい"
Phrase: "夏を爽やかに過ごしたい"
Score: 1.0 (perfect match)
```

This hybrid approach ensures that:
- Completions that start with your exact input are prioritized
- Semantically relevant phrases (containing your key words) rank higher
- Shorter, more relevant completions aren't penalized unfairly

## Security Considerations

**Important: Pickle Security Warning**

N-gram models are serialized using Python's `pickle` module. **Pickle files can execute arbitrary code when loaded.**

- ⚠️ Only load model files from trusted sources
- ⚠️ Do not load `.pkl` files from unknown or untrusted origins
- ⚠️ Loading custom models will display a security warning

```python
# Safe: Using default model (included with package)
completer = JaCompleter()

# Warning: Loading custom model (only use trusted files!)
completer.load_ngram_model("custom_model.pkl")  # Shows security warning
```

For more details, see [DEVELOPING.md](DEVELOPING.md#セキュリティ).

## Building Custom N-gram Model

For advanced users who want to build their own N-gram model:

```bash
# Download Japanese Wikipedia dump
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2

# Extract text using WikiExtractor
python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml.bz2 -o wiki_text/

# Build N-gram model
python scripts/build_ngram_model.py --input wiki_text/ --output my_model.pkl --verbose
```

## Contributing

Contributions are welcome! Please see [DEVELOPING.md](DEVELOPING.md) for development setup and guidelines.

## License

MIT
