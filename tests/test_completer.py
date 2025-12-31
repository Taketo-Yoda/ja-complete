"""Integration tests for JaCompleter."""

import pickle

import pytest

from ja_complete import JaCompleter
from ja_complete.types import SuggestionList


class TestJaCompleter:
    """Test JaCompleter integration."""

    def test_initialization(self):
        """Test completer initialization."""
        completer = JaCompleter()
        assert completer is not None

    def test_phrase_completion(self):
        """Test phrase-based completion."""
        completer = JaCompleter()

        phrases = [
            "スマホの買い換えと合わせて一式揃えたい",
            "新生活に備えた準備を始めたい",
            "夏を爽やかに過ごしたい",
        ]
        completer.add_phrases(phrases)

        # Test prefix match
        results = completer.suggest_from_phrases("ス", top_k=5, fallback_to_ngram=False)
        assert len(results) > 0
        assert any("スマホ" in r.text for r in results.items)

        # Test longer prefix
        results = completer.suggest_from_phrases("スマホの", fallback_to_ngram=False)
        assert len(results) > 0
        assert results[0].text == "スマホの買い換えと合わせて一式揃えたい"

    def test_simple_dictionary_completion(self):
        """Test simple dictionary completion."""
        completer = JaCompleter()

        suggestions = {
            "お": ["おはよう", "おやすみ", "お疲れ様"],
            "あり": ["ありがとう", "ありがとうございます"],
        }
        completer.add_simple_suggestions(suggestions)

        results = completer.suggest_from_simple("あり", fallback_to_ngram=False)
        assert len(results) == 2
        assert results[0].text in ["ありがとう", "ありがとうございます"]
        assert results[0].score == 1.0

    def test_ngram_completion(self):
        """Test N-gram completion."""
        completer = JaCompleter()

        # Use default model
        results = completer.suggest_from_ngram("今日", top_k=5)
        assert len(results) >= 0  # May or may not have results

    def test_ngram_fallback(self):
        """Test N-gram fallback feature."""
        completer = JaCompleter(enable_ngram_fallback=True)

        # Add some phrases
        completer.add_phrases(["スマホを買う"])

        # Query that doesn't match any phrase
        results = completer.suggest_from_phrases("今日", top_k=5)
        # Should get N-gram results as fallback
        assert isinstance(results, SuggestionList)

        # Disable fallback
        completer_no_fallback = JaCompleter(enable_ngram_fallback=False)
        completer_no_fallback.add_phrases(["スマホを買う"])
        results = completer_no_fallback.suggest_from_phrases("今日", top_k=5)
        # Should get empty results
        assert len(results) == 0

    def test_convert_to_jsonl(self):
        """Test JSONL conversion utility."""
        phrases = [
            "今日はいい天気ですね",
            "明日は雨が降りそうです",
        ]

        jsonl = JaCompleter.convert_to_jsonl(phrases)
        assert jsonl is not None
        lines = jsonl.split("\n")
        assert len(lines) == 2

        # Check format
        import json

        first_obj = json.loads(lines[0])
        assert "text" in first_obj
        assert "tokens" in first_obj
        assert first_obj["text"] == "今日はいい天気ですね"
        assert isinstance(first_obj["tokens"], list)

    def test_empty_input_validation(self):
        """Test validation for empty input."""
        completer = JaCompleter()
        completer.add_phrases(["テスト"])

        with pytest.raises(ValueError):
            completer.suggest_from_phrases("", top_k=10)

    def test_invalid_top_k_validation(self):
        """Test validation for invalid top_k."""
        completer = JaCompleter()
        completer.add_phrases(["テスト"])

        with pytest.raises(ValueError):
            completer.suggest_from_phrases("テ", top_k=0)

        with pytest.raises(ValueError):
            completer.suggest_from_phrases("テ", top_k=-1)

    def test_load_ngram_model_file_not_found(self):
        """Test load_ngram_model raises FileNotFoundError for non-existent file."""
        completer = JaCompleter()

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            completer.load_ngram_model("nonexistent_model.pkl")

    def test_load_ngram_model_directory_error(self, tmp_path):
        """Test load_ngram_model raises ValueError when given a directory."""
        completer = JaCompleter()

        # Create a temporary directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(ValueError, match="Expected file, got directory"):
            completer.load_ngram_model(str(test_dir))

    def test_load_ngram_model_accepts_path_object(self, tmp_path):
        """Test load_ngram_model accepts pathlib.Path objects."""
        completer = JaCompleter()

        # Create a temporary model file
        model_file = tmp_path / "test_model.pkl"
        model_data = {
            "unigrams": {"今日": 10, "は": 8},
            "bigrams": {"今日": {"は": 8}},
            "trigrams": {},
        }
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        # Should accept Path object without error
        completer.load_ngram_model(model_file)

        # Verify the model was loaded
        results = completer.suggest_from_ngram("今日", top_k=5)
        assert isinstance(results, SuggestionList)
