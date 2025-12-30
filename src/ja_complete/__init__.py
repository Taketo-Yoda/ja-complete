"""ja-complete: Lightweight offline Japanese input completion library.

This library provides multiple independent completion methods for Japanese text:
- Phrase-based completion with automatic prefix generation
- N-gram statistical completion
- Simple dictionary completion

All methods work offline without requiring LLMs or databases.
"""

from ja_complete.completer import JaCompleter

__version__ = "0.1.0"
__all__ = ["JaCompleter"]
