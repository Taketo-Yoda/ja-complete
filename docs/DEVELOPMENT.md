# ja-complete Development Guide

This document is intended for Claude Code and developers working on the ja-complete library.

## Project Overview

`ja-complete` is a lightweight, offline Japanese text completion library that provides multiple independent completion methods without requiring LLMs or databases.

## Architecture Design

### Core Principles

1. **Multiple Independent APIs**: Each completion method (phrases, N-gram, simple dictionary, JSONL) is exposed as a separate method
2. **Morphological Analysis**: Uses Janome for Japanese text tokenization
3. **Offline First**: All functionality works without internet connection
4. **Lightweight**: Minimal dependencies, small package size

### Module Structure

```
src/ja_complete/
├── __init__.py              # Public API exports
├── completer.py             # Main JaCompleter class
├── tokenizer.py             # Janome wrapper for morphological analysis
├── models/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for completion models
│   ├── phrase.py            # Phrase-based completion model
│   ├── ngram.py             # N-gram statistical model
│   └── simple.py            # Simple dictionary model
├── data/
│   └── default_ngram.pkl    # Default N-gram model (minimal/dummy)
└── cli.py                   # CLI entry point
```

## Technical Specifications

### 1. Tokenizer Module (`tokenizer.py`)

Wrapper around Janome tokenizer with convenient methods.

**Functions:**

```python
def tokenize(text: str) -> list[str]:
    """
    Tokenize Japanese text into morphemes.

    Args:
        text: Japanese text to tokenize

    Returns:
        List of surface forms (morphemes)
    """

def get_morphemes(text: str) -> list[dict[str, str]]:
    """
    Get detailed morpheme information.

    Args:
        text: Japanese text to analyze

    Returns:
        List of dicts with keys: surface, pos (part of speech), base_form
    """

def extract_bunsetsu(text: str) -> list[str]:
    """
    Extract bunsetsu (phrase chunks) using rule-based approach.

    Rule: 自立語 (content word) + 付属語 (function words)

    Implementation follows these rules:
    1. After particles (助詞) - except when followed by another particle
    2. After auxiliary verbs (助動詞)
    3. Before independent words (名詞、動詞、形容詞、副詞、連体詞、接続詞)
       - But not if current morpheme is prefix/suffix (接頭詞、接尾辞)

    Args:
        text: Japanese text

    Returns:
        List of bunsetsu (phrase chunks)

    Example:
        "今日はいい天気ですね" → ["今日は", "いい", "天気ですね"]
    """
```

**Detailed Bunsetsu Extraction Algorithm:**

```python
def extract_bunsetsu(text: str, tokenizer) -> list[str]:
    """Extract bunsetsu using rule-based approach."""
    morphemes = tokenizer.get_morphemes(text)

    bunsetsu_list = []
    current_bunsetsu = ""

    for i, morph in enumerate(morphemes):
        surface = morph['surface']
        pos = morph['pos']  # Main POS category

        current_bunsetsu += surface

        # Check if we should end current bunsetsu
        should_break = False

        # Rule 1: After particles (助詞) - except when followed by another particle
        if pos == '助詞':
            next_pos = morphemes[i + 1]['pos'] if i + 1 < len(morphemes) else None
            if next_pos != '助詞':
                should_break = True

        # Rule 2: After auxiliary verbs (助動詞)
        elif pos == '助動詞':
            should_break = True

        # Rule 3: Before independent words (自立語)
        elif i + 1 < len(morphemes):
            next_pos = morphemes[i + 1]['pos']
            if next_pos in ['名詞', '動詞', '形容詞', '副詞', '連体詞', '接続詞']:
                # But not if current is prefix/suffix
                if pos not in ['接頭詞', '接尾辞']:
                    should_break = True

        if should_break or i == len(morphemes) - 1:
            if current_bunsetsu:
                bunsetsu_list.append(current_bunsetsu)
                current_bunsetsu = ""

    return bunsetsu_list
```

**Implementation Notes:**
- Use `janome.tokenizer.Tokenizer` as base
- Cache tokenizer instance as module-level singleton
- Handle encoding properly for Japanese text
- Bunsetsu rules may need adjustment based on real-world usage patterns

### 2. Base Model (`models/base.py`)

Abstract base class defining the interface for all completion models.

```python
from abc import ABC, abstractmethod
from typing import Any

class CompletionModel(ABC):
    """Base class for all completion models."""

    @abstractmethod
    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Generate completions for input text.

        Args:
            input_text: User input text
            top_k: Maximum number of suggestions to return

        Returns:
            List of dicts with keys: 'text' (completion), 'score' (float 0-1)
            Sorted by score descending
        """
        pass
```

### 3. Phrase-based Model (`models/phrase.py`)

Generates completions from a custom phrase list using morphological analysis.

**Class: `PhraseModel(CompletionModel)`**

```python
class PhraseModel(CompletionModel):
    def __init__(self):
        self.phrases: set[str] = set()
        self.prefix_map: dict[str, set[str]] = {}

    def add_phrases(self, phrases: list[str]) -> None:
        """
        Add phrases and build prefix index.

        For each phrase:
        1. Tokenize using Janome
        2. Generate prefixes using multiple strategies:
           a. Character-level: First 1, 2, 3 characters
           b. Morpheme boundaries: After each morpheme
           c. Bunsetsu boundaries: After each bunsetsu (using extract_bunsetsu)
        3. Map prefix -> full phrase

        Example prefix generation:
        Phrase: "スマホの買い換えと合わせて一式揃えたい"

        Character-level:
        - "ス"
        - "スマ"
        - "スマホ"

        Morpheme boundaries (from Janome):
        - "スマホ"
        - "スマホの"
        - "スマホの買い換え"
        - "スマホの買い換えと"
        - ... (each morpheme boundary)

        Bunsetsu boundaries (rule-based):
        - "スマホの"  (after particle)
        - "買い換えと"  (after particle)
        - "合わせて"  (after auxiliary verb)
        - ... (each bunsetsu boundary)

        All unique prefixes are indexed to point to this phrase.
        """

    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Find phrases matching input_text as prefix.

        Uses hybrid scoring algorithm combining:
        1. Prefix matching quality (60% weight)
        2. Morpheme overlap (40% weight)

        Return top_k results sorted by score descending.
        """
```

**Scoring Algorithm Details:**

The scoring algorithm combines two components to provide better completion ranking:

**Component 1: Prefix Matching Score (0.0 ~ 0.6)**

```python
if phrase == input_text:
    prefix_score = 0.6  # Perfect match
elif phrase.startswith(input_text):
    ratio = len(input_text) / len(phrase)
    prefix_score = 0.3 + (ratio * 0.3)  # Range: 0.3 ~ 0.6
else:
    return 0.0  # No match - skip this phrase
```

**Component 2: Morpheme Overlap Score (0.0 ~ 0.4)**

```python
input_morphemes = set(tokenizer.tokenize(input_text))
phrase_morphemes = set(tokenizer.tokenize(phrase))

if not input_morphemes:
    morpheme_score = 0.0
else:
    # Calculate Jaccard similarity for morphemes
    intersection = input_morphemes & phrase_morphemes
    morpheme_overlap_ratio = len(intersection) / len(input_morphemes)
    morpheme_score = morpheme_overlap_ratio * 0.4
```

**Total Score Calculation:**

```python
total_score = prefix_score + morpheme_score
return min(total_score, 1.0)  # Cap at 1.0
```

**Scoring Examples:**

Example 1: Long prefix match with full morpheme overlap
```
Input: "スマホの買い換え" (8 chars, 3 morphemes: スマホ, の, 買い換え)
Phrase: "スマホの買い換えと合わせて一式揃えたい" (20 chars)

Prefix score: 0.3 + (8/20 * 0.3) = 0.3 + 0.12 = 0.42
Morpheme score: 3/3 * 0.4 = 0.4
Total: 0.82
```

Example 2: Short prefix with full morpheme overlap
```
Input: "スマホ" (3 chars, 1 morpheme: スマホ)
Phrase: "スマホの買い換えと合わせて一式揃えたい" (20 chars)

Prefix score: 0.3 + (3/20 * 0.3) = 0.345
Morpheme score: 1/1 * 0.4 = 0.4
Total: 0.745
```

Example 3: Perfect match
```
Input: "スマホの買い換えと合わせて一式揃えたい"
Phrase: "スマホの買い換えと合わせて一式揃えたい"

Prefix score: 0.6
Morpheme score: All morphemes match = 0.4
Total: 1.0
```

**Weight Configuration:**

The default weights are:
- `PREFIX_WEIGHT = 0.6` (prefix matching quality)
- `MORPHEME_WEIGHT = 0.4` (semantic similarity via morphemes)

These can be adjusted based on use case:
- For strict prefix completion: increase PREFIX_WEIGHT (e.g., 0.7/0.3)
- For semantic matching: increase MORPHEME_WEIGHT (e.g., 0.5/0.5)
```

### 4. N-gram Model (`models/ngram.py`)

Statistical completion using bigram/trigram probabilities.

**Class: `NgramModel(CompletionModel)`**

```python
class NgramModel(CompletionModel):
    def __init__(self, model_path: str | None = None):
        """
        Initialize N-gram model.

        Args:
            model_path: Path to pickled model file.
                       If None, load default model from data/default_ngram.pkl
        """
        self.bigrams: dict[str, dict[str, int]] = {}
        self.trigrams: dict[tuple[str, str], dict[str, int]] = {}
        self.unigrams: dict[str, int] = {}

        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()

    def load_model(self, path: str) -> None:
        """Load pickled N-gram model."""

    def load_default_model(self) -> None:
        """Load default model from package data."""

    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Predict next words using N-gram probabilities.

        1. Tokenize input_text
        2. Get last 1-2 tokens as context
        3. Use trigram if available, fallback to bigram
        4. Calculate probabilities with Laplace smoothing (alpha=1.0)
        5. Generate completions by:
           - Taking top probable next tokens
           - Appending to input_text
        6. Score = probability (0-1)
        """
```

**Model File Format:**
```python
{
    'unigrams': {'token': count, ...},
    'bigrams': {'token1': {'token2': count, ...}, ...},
    'trigrams': {('token1', 'token2'): {'token3': count, ...}, ...}
}
```

### 5. Simple Dictionary Model (`models/simple.py`)

Direct prefix-to-suggestions mapping.

**Class: `SimpleDictModel(CompletionModel)`**

```python
class SimpleDictModel(CompletionModel):
    def __init__(self):
        self.suggestions: dict[str, list[str]] = {}

    def add_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """
        Add or update prefix mappings.

        Args:
            suggestions: Dict mapping prefix -> list of completions
        """

    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Return exact prefix matches.

        1. Look up input_text in suggestions dict
        2. Return matches with score=1.0
        3. If no exact match, try progressively shorter prefixes
        4. Return up to top_k results
        """
```

### 6. Main Completer (`completer.py`)

Central class that manages all completion models.

**Class: `JaCompleter`**

```python
from ja_complete.models.phrase import PhraseModel
from ja_complete.models.ngram import NgramModel
from ja_complete.models.simple import SimpleDictModel

class JaCompleter:
    """Main completion class providing multiple independent completion methods."""

    def __init__(self, enable_ngram_fallback: bool = True):
        """
        Initialize JaCompleter.

        Args:
            enable_ngram_fallback: If True, phrase-based and simple dictionary
                                  completions will fall back to N-gram when no
                                  matches are found (default: True)
        """
        self._phrase_model = PhraseModel()
        self._ngram_model = NgramModel()  # Loads default model
        self._simple_model = SimpleDictModel()
        self._enable_ngram_fallback = enable_ngram_fallback

    # Phrase-based methods
    def add_phrases(self, phrases: list[str]) -> None:
        """Add phrases to phrase-based completion."""
        self._phrase_model.add_phrases(phrases)

    def suggest_from_phrases(
        self,
        input_text: str,
        top_k: int = 10,
        fallback_to_ngram: bool | None = None
    ) -> list[dict[str, Any]]:
        """
        Get completions from phrase model with optional N-gram fallback.

        Args:
            input_text: User input text
            top_k: Maximum number of suggestions
            fallback_to_ngram: Override default fallback behavior.
                             If None, uses instance setting.

        Returns:
            List of completion dicts with 'text' and 'score' keys

        Behavior:
            1. Try to get completions from phrase model
            2. If no matches and fallback is enabled, use N-gram model
            3. Return top_k results sorted by score
        """
        results = self._phrase_model.suggest(input_text, top_k)

        # Fallback to N-gram if enabled and no results
        use_fallback = (fallback_to_ngram
                       if fallback_to_ngram is not None
                       else self._enable_ngram_fallback)

        if not results and use_fallback:
            results = self._ngram_model.suggest(input_text, top_k)

        return results

    # N-gram methods
    def load_ngram_model(self, model_path: str) -> None:
        """Load custom N-gram model."""
        self._ngram_model = NgramModel(model_path)

    def suggest_from_ngram(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Get completions from N-gram model only."""
        return self._ngram_model.suggest(input_text, top_k)

    # Simple dictionary methods
    def add_simple_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """Add simple prefix-to-completion mappings."""
        self._simple_model.add_suggestions(suggestions)

    def suggest_from_simple(
        self,
        input_text: str,
        top_k: int = 10,
        fallback_to_ngram: bool | None = None
    ) -> list[dict[str, Any]]:
        """
        Get completions from simple dictionary with optional N-gram fallback.

        Args:
            input_text: User input text
            top_k: Maximum number of suggestions
            fallback_to_ngram: Override default fallback behavior.
                             If None, uses instance setting.

        Returns:
            List of completion dicts with 'text' and 'score' keys

        Behavior:
            1. Try to get completions from simple dictionary
            2. If no matches and fallback is enabled, use N-gram model
            3. Return top_k results sorted by score
        """
        results = self._simple_model.suggest(input_text, top_k)

        # Fallback to N-gram if enabled and no results
        use_fallback = (fallback_to_ngram
                       if fallback_to_ngram is not None
                       else self._enable_ngram_fallback)

        if not results and use_fallback:
            results = self._ngram_model.suggest(input_text, top_k)

        return results

    # Utility methods
    @staticmethod
    def convert_to_jsonl(phrases: list[str]) -> str:
        """
        Convert list of phrases to JSONL format for N-gram model training.

        Each phrase becomes a JSON object with metadata that can be used
        for building custom N-gram models or training data.

        Args:
            phrases: List of Japanese phrases

        Returns:
            JSONL string (one JSON object per line)

        Example:
            >>> phrases = ["今日はいい天気", "明日は雨"]
            >>> jsonl = JaCompleter.convert_to_jsonl(phrases)
            >>> print(jsonl)
            {"text": "今日はいい天気", "tokens": ["今日", "は", "いい", "天気"]}
            {"text": "明日は雨", "tokens": ["明日", "は", "雨"]}

        Note:
            The JSONL output includes tokenized versions of phrases for
            easier N-gram model training.
        """
        import json
        from ja_complete.tokenizer import tokenize

        lines = []
        for phrase in phrases:
            tokens = tokenize(phrase)
            obj = {
                "text": phrase,
                "tokens": tokens
            }
            lines.append(json.dumps(obj, ensure_ascii=False))

        return "\n".join(lines)
```

### 7. CLI Module (`cli.py`)

Command-line interface with subcommands for each completion method.

```python
import argparse
import json
from ja_complete import JaCompleter

def main() -> None:
    parser = argparse.ArgumentParser(description="Japanese text completion")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Phrase subcommand
    phrase_parser = subparsers.add_parser('phrase', help='Phrase-based completion')
    phrase_parser.add_argument('input', help='Input text')
    phrase_parser.add_argument('--phrases', help='File with phrases (one per line)')
    phrase_parser.add_argument('--top-k', type=int, default=10)

    # N-gram subcommand
    ngram_parser = subparsers.add_parser('ngram', help='N-gram completion')
    ngram_parser.add_argument('input', help='Input text')
    ngram_parser.add_argument('--model', help='Custom model path')
    ngram_parser.add_argument('--top-k', type=int, default=10)

    # Simple subcommand
    simple_parser = subparsers.add_parser('simple', help='Simple dictionary completion')
    simple_parser.add_argument('input', help='Input text')
    simple_parser.add_argument('--dict', help='JSON file with suggestions')
    simple_parser.add_argument('--top-k', type=int, default=10)

    args = parser.parse_args()
    completer = JaCompleter()

    # Execute appropriate subcommand
    # ... implementation

    # Print results as JSON
    results = completer.suggest_from_X(args.input, args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))
```

### 8. Build N-gram Model Script (`scripts/build_ngram_model.py`)

Script to build N-gram model from text corpus.

```python
import argparse
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from ja_complete.tokenizer import tokenize

def build_ngram_model(input_dir: Path, output_path: Path) -> None:
    """
    Build N-gram model from text files.

    Args:
        input_dir: Directory containing text files
        output_path: Path to save pickled model
    """
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    # Process all .txt files in input_dir
    for text_file in input_dir.rglob('*.txt'):
        with open(text_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = tokenize(line.strip())

                # Count unigrams
                unigrams.update(tokens)

                # Count bigrams
                for i in range(len(tokens) - 1):
                    bigrams[tokens[i]][tokens[i + 1]] += 1

                # Count trigrams
                for i in range(len(tokens) - 2):
                    trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

    # Save model
    model = {
        'unigrams': dict(unigrams),
        'bigrams': {k: dict(v) for k, v in bigrams.items()},
        'trigrams': {k: dict(v) for k, v in trigrams.items()}
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

def main() -> None:
    parser = argparse.ArgumentParser(description='Build N-gram model')
    parser.add_argument('--input', required=True, help='Input directory with text files')
    parser.add_argument('--output', required=True, help='Output model path (.pkl)')

    args = parser.parse_args()
    build_ngram_model(Path(args.input), Path(args.output))
    print(f"Model saved to {args.output}")

if __name__ == '__main__':
    main()
```

## Implementation Guidelines

### Design Principles

**This project follows Domain-Driven Design (DDD), SOLID principles, and applies Gang of Four (GoF) design patterns where appropriate.**

1. **Domain-Driven Design (DDD)**:
   - Organize code around domain concepts (Completion, Tokenization, Scoring)
   - Use ubiquitous language in code and documentation
   - Separate domain logic from infrastructure concerns

2. **SOLID Principles**:
   - **S**ingle Responsibility: Each class has one reason to change
   - **O**pen/Closed: Open for extension, closed for modification (use abstract base classes)
   - **L**iskov Substitution: All CompletionModel implementations are interchangeable
   - **I**nterface Segregation: Clients depend only on methods they use
   - **D**ependency Inversion: Depend on abstractions (CompletionModel), not concretions

3. **Design Patterns (Apply where appropriate)**:
   - **Strategy Pattern**: CompletionModel implementations (phrase, N-gram, simple)
   - **Singleton Pattern**: Tokenizer instance (module-level)
   - **Factory Pattern**: Model creation if needed
   - **Decorator Pattern**: Adding caching or logging to completion methods
   - **Template Method**: Base scoring algorithm with customizable steps

### Code Style

1. **Type Hints**: Use strict typing everywhere
   - Use `list[str]` instead of `List[str]` (Python 3.9+)
   - Use `dict[str, Any]` instead of `Dict[str, Any]`
   - Use `tuple[str, str]` instead of `Tuple[str, str]`

2. **Error Handling**:
   - Raise `ValueError` for invalid input
   - Raise `FileNotFoundError` for missing model files
   - Use custom exceptions if needed
   - Create domain-specific exceptions (e.g., `ModelNotLoadedError`, `InvalidPhraseError`)

3. **Validation**:
   - Use Pydantic for data validation where appropriate
   - Validate `top_k > 0`
   - Validate non-empty strings

4. **Documentation**:
   - Use Google-style docstrings
   - Include type hints in docstrings
   - Provide usage examples in docstrings

### Testing Requirements

All tests should be in the `tests/` directory.

**Required test files:**

1. `test_tokenizer.py`: Test morphological analysis
   - Test basic tokenization
   - Test bunsetsu extraction
   - Test edge cases (empty string, single character, etc.)

2. `test_phrase.py`: Test phrase-based completion
   - Test prefix matching
   - Test scoring algorithm
   - Test with Japanese text

3. `test_ngram.py`: Test N-gram model
   - Test model loading
   - Test probability calculation
   - Test completion generation

4. `test_simple.py`: Test simple dictionary
   - Test exact prefix match
   - Test partial prefix match

5. `test_completer.py`: Integration tests
   - Test all public methods
   - Test multiple completion sources
   - Test error handling

6. `test_cli.py`: CLI tests
   - Test each subcommand
   - Test argument parsing

### Default N-gram Model

For initial development, create a minimal dummy model:

```python
# src/ja_complete/data/create_dummy_model.py
import pickle
from pathlib import Path

dummy_model = {
    'unigrams': {'今日': 10, 'は': 20, 'いい': 5, '天気': 5},
    'bigrams': {
        '今日': {'は': 10},
        'は': {'いい': 5},
        'いい': {'天気': 5}
    },
    'trigrams': {
        ('今日', 'は'): {'いい': 5},
        ('は', 'いい'): {'天気': 5}
    }
}

output_path = Path(__file__).parent / 'default_ngram.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(dummy_model, f)
```

## Development Workflow

1. **Setup**:
   ```bash
   uv sync --group dev
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Lint and format**:
   ```bash
   ruff check .
   ruff format .
   ```

4. **Build package**:
   ```bash
   uv build
   ```

5. **Install locally for testing**:
   ```bash
   uv pip install -e .
   ```

## Open Questions / Future Considerations

1. **JSONL support**: Define JSONL format and implement loading
2. **Model compression**: Consider compression for larger N-gram models
3. **Unicode normalization**: Should we normalize Japanese text (NFKC)?
4. **Performance optimization**: Profile and optimize hot paths
5. **Incremental updates**: Support adding phrases/suggestions without rebuilding indices

## Implementation Order (Recommendation)

1. Start with `tokenizer.py` - foundation for everything
2. Implement `models/base.py` - defines interface
3. Implement `models/simple.py` - simplest model, good for testing
4. Implement `models/phrase.py` - core functionality
5. Create dummy N-gram model
6. Implement `models/ngram.py`
7. Implement `completer.py` - ties everything together
8. Implement `cli.py`
9. Write comprehensive tests
10. Implement `scripts/build_ngram_model.py`
