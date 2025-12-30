# ja-complete Architecture

## Design Philosophy

The ja-complete library follows these core design principles:

1. **Simplicity**: Minimize complexity while providing powerful features
2. **Modularity**: Each completion method is independent and self-contained
3. **Offline-first**: No external dependencies or network calls required
4. **Extensibility**: Easy to add new completion methods or models

### Software Design Principles

**Domain-Driven Design (DDD)**

The codebase is organized around domain concepts:
- **Completion Domain**: Core business logic for text completion
- **Tokenization Domain**: Japanese text analysis and morpheme extraction
- **Scoring Domain**: Algorithms for ranking completion candidates
- **Model Domain**: Statistical models and phrase storage

**SOLID Principles**

1. **Single Responsibility Principle (SRP)**:
   - `PhraseModel`: Only handles phrase-based completion
   - `NgramModel`: Only handles N-gram statistical completion
   - `Tokenizer`: Only handles text tokenization

2. **Open/Closed Principle (OCP)**:
   - `CompletionModel` abstract base class allows new completion strategies without modifying existing code
   - New models can be added by extending `CompletionModel`

3. **Liskov Substitution Principle (LSP)**:
   - All `CompletionModel` implementations can be used interchangeably
   - JaCompleter depends on the abstract interface, not concrete implementations

4. **Interface Segregation Principle (ISP)**:
   - Each model exposes only the methods it needs (e.g., `add_phrases` only in `PhraseModel`)
   - Clients aren't forced to depend on methods they don't use

5. **Dependency Inversion Principle (DIP)**:
   - High-level `JaCompleter` depends on `CompletionModel` abstraction
   - Low-level models implement the abstraction
   - Dependencies flow inward (toward abstractions)

**Gang of Four (GoF) Design Patterns**

Applied patterns in this project:

1. **Strategy Pattern**:
   - Different completion strategies (phrase, N-gram, simple) implement common interface
   - Allows runtime selection of completion strategy

2. **Facade Pattern**:
   - `JaCompleter` provides simplified interface to complex subsystem
   - Hides complexity of multiple models and tokenizer

3. **Singleton Pattern**:
   - Tokenizer instance is cached as module-level singleton
   - Avoids repeated initialization overhead

4. **Template Method Pattern** (potential):
   - Base scoring algorithm in `CompletionModel` with customizable steps
   - Subclasses override specific scoring behaviors

5. **Decorator Pattern** (potential):
   - Can wrap completion methods with caching or logging
   - Non-invasive enhancement of functionality

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                     JaCompleter                         │
│  (Main API - orchestrates multiple completion models)   │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Phrase   │   │  N-gram   │   │  Simple   │
    │   Model   │   │   Model   │   │   Dict    │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                    ┌───────────┐
                    │ Tokenizer │
                    │ (Janome)  │
                    └───────────┘
```

### Component Responsibilities

#### 1. JaCompleter (Facade)
- **Purpose**: Unified interface for all completion methods
- **Responsibilities**:
  - Initialize and manage completion models
  - Expose independent completion methods
  - Handle configuration
- **Dependencies**: All model classes

#### 2. CompletionModel (Abstract Base)
- **Purpose**: Define common interface for all models
- **Responsibilities**:
  - Enforce consistent API across models
  - Define common types and return formats
- **Dependencies**: None

#### 3. PhraseModel
- **Purpose**: Provide phrase-based completion
- **Responsibilities**:
  - Build prefix index from phrases
  - Match user input to phrases
  - Score matches by prefix length
- **Dependencies**: Tokenizer
- **Data structures**:
  - `phrases: set[str]` - All registered phrases
  - `prefix_map: dict[str, set[str]]` - Prefix → phrases mapping

#### 4. NgramModel
- **Purpose**: Provide statistical completion using N-grams
- **Responsibilities**:
  - Load and manage N-gram statistics
  - Calculate continuation probabilities
  - Generate completions with scores
- **Dependencies**: Tokenizer
- **Data structures**:
  - `unigrams: dict[str, int]` - Token counts
  - `bigrams: dict[str, dict[str, int]]` - Bigram counts
  - `trigrams: dict[tuple[str, str], dict[str, int]]` - Trigram counts

#### 5. SimpleDictModel
- **Purpose**: Simple prefix-based lookup
- **Responsibilities**:
  - Store prefix → suggestions mapping
  - Perform exact and fuzzy prefix matching
- **Dependencies**: None
- **Data structures**:
  - `suggestions: dict[str, list[str]]` - Prefix → completions

#### 6. Tokenizer
- **Purpose**: Japanese text analysis
- **Responsibilities**:
  - Tokenize Japanese text into morphemes
  - Extract bunsetsu (phrase chunks)
  - Provide part-of-speech information
- **Dependencies**: Janome

## Data Flow

### Phrase-based Completion Flow (with N-gram Fallback)

```
User Input ("ス")
    │
    ▼
JaCompleter.suggest_from_phrases()
    │
    ▼
PhraseModel.suggest()
    │
    ├─→ Normalize input
    │
    ├─→ Look up in prefix_map
    │
    ├─→ Calculate scores (hybrid: prefix 60% + morpheme 40%)
    │
    ├─→ Sort by score
    │
    └─→ Return results
         │
         ▼
    Has results?
         │
         ├─ Yes ──→ Return phrase completions
         │
         └─ No + fallback enabled?
              │
              └─ Yes ──→ NgramModel.suggest()
                            │
                            └─→ Return N-gram completions

Output: [{'text': 'スマホの...', 'score': 0.82}, ...]
        or N-gram results if no phrase matches
```

**Note**: Simple dictionary completion follows the same fallback pattern.

### N-gram Completion Flow

```
User Input ("今日は")
    │
    ▼
JaCompleter.suggest_from_ngram()
    │
    ▼
NgramModel.suggest()
    │
    ├─→ Tokenize input → ['今日', 'は']
    │
    ├─→ Get context (last 1-2 tokens)
    │
    ├─→ Look up in trigrams/bigrams
    │
    ├─→ Calculate probabilities (with smoothing)
    │
    ├─→ Generate completions
    │
    ├─→ Sort by probability
    │
    └─→ Return top_k results

Output: [{'text': '今日はいい天気', 'score': 0.85}, ...]
```

## Design Decisions

### Why Independent Methods Instead of Unified API?

**Decision**: Provide separate methods (`suggest_from_phrases()`, `suggest_from_ngram()`, etc.) rather than a single `suggest()` method with flags.

**Rationale**:
1. **Clarity**: Users explicitly choose which completion source to use
2. **Flexibility**: Easy to use multiple sources and combine results in application code
3. **Performance**: No overhead from checking/combining multiple sources
4. **Simplicity**: Each method has clear, focused behavior

**Trade-off**: Slightly more verbose API, but more predictable and easier to understand.

### Why N-gram Fallback for Phrase and Simple Dictionary?

**Decision**: Phrase-based and simple dictionary completion methods automatically fall back to N-gram when no matches are found (configurable).

**Rationale**:
1. **Better UX**: Always provides suggestions, even for unregistered inputs
2. **Graceful degradation**: Prioritizes custom data but doesn't fail silently
3. **Flexibility**: Users can disable fallback for strict matching scenarios
4. **Practical**: Most real-world use cases benefit from "some suggestion" vs "no suggestion"

**Implementation**:
```python
# Default: fallback enabled
completer = JaCompleter()  # enable_ngram_fallback=True
results = completer.suggest_from_phrases("未登録")  # Returns N-gram results

# Strict mode: no fallback
completer = JaCompleter(enable_ngram_fallback=False)
results = completer.suggest_from_phrases("未登録")  # Returns []

# Per-call override
results = completer.suggest_from_phrases("未登録", fallback_to_ngram=False)
```

**Trade-offs**:
- Adds slight complexity to API (extra parameter)
- N-gram results may be less relevant than phrase matches
- Benefit: Significantly improves completion coverage

### Why Src Layout?

**Decision**: Use `src/ja_complete/` instead of `ja_complete/` at root.

**Rationale**:
1. **Best practice** for PyPI packages
2. **Prevents accidental imports** from source directory during development
3. **Forces proper installation** for testing
4. **Cleaner separation** between source and other files

### Why Pickle for N-gram Model?

**Decision**: Use Python's `pickle` for serializing N-gram models.

**Rationale**:
1. **Simplicity**: Built-in, no extra dependencies
2. **Performance**: Fast serialization/deserialization
3. **Python-native**: Natural fit for Python data structures

**Trade-offs**:
- Not human-readable (use JSON for small models if needed)
- Python version compatibility concerns (use protocol 4+)
- Security concerns (only load trusted models)

**Alternatives considered**:
- JSON: Too slow for large models, larger file size
- MessagePack: Extra dependency
- SQLite: Overkill for simple key-value storage

### Why Janome?

**Decision**: Use Janome for morphological analysis.

**Rationale**:
1. **Pure Python**: Easy to install, no C dependencies
2. **Offline**: Dictionary included in package
3. **Sufficient accuracy**: Good enough for completion use case
4. **Lightweight**: Smaller than MeCab alternatives

**Trade-offs**:
- Slower than MeCab
- Less accurate than neural models

**Alternatives considered**:
- fugashi/MeCab: Requires C dependencies, harder to install
- SudachiPy: Larger package, more complex
- transformers: Too heavy, requires GPU for good performance

## Extension Points

### Adding New Completion Models

To add a new completion model:

1. Create new file in `src/ja_complete/models/`
2. Inherit from `CompletionModel`
3. Implement `suggest()` method
4. Add corresponding methods to `JaCompleter`

Example:

```python
# src/ja_complete/models/fuzzy.py
from ja_complete.models.base import CompletionModel

class FuzzyModel(CompletionModel):
    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        # Fuzzy matching implementation
        pass

# src/ja_complete/completer.py
class JaCompleter:
    def __init__(self):
        # ...
        self._fuzzy_model = FuzzyModel()

    def suggest_from_fuzzy(self, input_text: str, top_k: int = 10):
        return self._fuzzy_model.suggest(input_text, top_k)
```

### Adding Custom Tokenizers

To support alternative tokenizers:

1. Create wrapper class with same interface as `tokenizer.py`
2. Add initialization parameter to models
3. Pass custom tokenizer instance

```python
completer = JaCompleter(tokenizer=MyCustomTokenizer())
```

## Performance Considerations

### Memory Usage

- **PhraseModel**: O(n × m) where n = number of phrases, m = average prefixes per phrase
- **NgramModel**: O(v²) for bigrams, O(v³) for trigrams where v = vocabulary size
- **SimpleDictModel**: O(k) where k = number of prefix entries

### Time Complexity

- **PhraseModel.suggest()**: O(log n) for prefix lookup, O(m) for scoring where m = matches
- **NgramModel.suggest()**: O(1) for context lookup, O(v) for probability calculation
- **SimpleDictModel.suggest()**: O(1) for exact match, O(k) for fuzzy matching

### Optimization Opportunities

1. **Phrase prefix indexing**: Use trie instead of dict for better prefix matching
2. **N-gram pruning**: Remove low-frequency N-grams to reduce memory
3. **Lazy loading**: Load models on-demand instead of at initialization
4. **Caching**: Cache tokenization results for repeated inputs

## Security Considerations

1. **Pickle safety**: Only load N-gram models from trusted sources
2. **Input validation**: Validate all user inputs (string length, character set)
3. **Resource limits**: Limit phrase list size, model file size
4. **Path traversal**: Validate file paths when loading models

## Future Enhancements

### Potential Features

1. **Incremental learning**: Update N-gram model with new text
2. **Multi-model ensemble**: Combine multiple sources with weighted scoring
3. **Context-aware completion**: Use surrounding text for better predictions
4. **Personalization**: Learn from user's completion selections
5. **JSONL support**: Load completions from structured data
6. **Unicode normalization**: Normalize Japanese text variants

### Backward Compatibility

When adding new features:
1. Keep existing method signatures unchanged
2. Add new optional parameters with defaults
3. Deprecate old features gradually with warnings
4. Document breaking changes clearly

## Testing Strategy

### Unit Tests
- Test each model independently
- Mock tokenizer for phrase/N-gram tests
- Test edge cases (empty input, very long input, special characters)

### Integration Tests
- Test JaCompleter with real models
- Test CLI with all subcommands
- Test model loading from files

### Performance Tests
- Benchmark completion speed
- Monitor memory usage
- Test with large phrase lists (10k+ items)
- Test with large N-gram models

### Test Data
- Use real Japanese text samples
- Include edge cases (emoji, Latin characters, numbers)
- Test with various text lengths
