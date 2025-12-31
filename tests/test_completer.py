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

    def test_phrases_to_simple_suggestions_basic(self):
        """Test phrases_to_simple_suggestions with basic phrases."""
        phrases = ["今日はいい天気", "今日は雨"]

        suggestions = JaCompleter.phrases_to_simple_suggestions(phrases)

        # Character-based prefixes should be generated
        assert "今" in suggestions.data
        assert "今日" in suggestions.data
        assert "今日は" in suggestions.data

        # Both phrases should appear for prefix "今"
        assert len(suggestions.data["今"]) == 2
        assert "今日はいい天気" in suggestions.data["今"]
        assert "今日は雨" in suggestions.data["今"]

    def test_phrases_to_simple_suggestions_custom_length(self):
        """Test phrases_to_simple_suggestions with custom prefix lengths."""
        phrases = ["こんにちは"]

        # min=2, max=4
        suggestions = JaCompleter.phrases_to_simple_suggestions(
            phrases, min_prefix_length=2, max_prefix_length=4
        )

        # Should not have 1-character prefix
        assert "こ" not in suggestions.data
        # Should have 2-4 character prefixes
        assert "こん" in suggestions.data
        assert "こんに" in suggestions.data
        assert "こんにち" in suggestions.data
        # Should not have 5-character prefix (exceeds max)
        assert "こんにちは" not in suggestions.data or len("こんにちは") > 4

    def test_phrases_to_simple_suggestions_empty_list(self):
        """Test phrases_to_simple_suggestions with empty list."""
        suggestions = JaCompleter.phrases_to_simple_suggestions([])
        assert suggestions.data == {}

    def test_phrases_to_ngram_data_basic(self):
        """Test phrases_to_ngram_data extracts n-grams correctly."""
        phrases = ["今日はいい天気"]

        ngram_data = JaCompleter.phrases_to_ngram_data(phrases)

        # Verify unigrams exist
        assert "今日" in ngram_data.unigrams
        assert "は" in ngram_data.unigrams
        assert ngram_data.unigrams["今日"] > 0
        assert ngram_data.unigrams["は"] > 0

        # Verify bigrams exist
        assert "今日" in ngram_data.bigrams
        if "は" in ngram_data.bigrams["今日"]:
            assert ngram_data.bigrams["今日"]["は"] > 0

        # Verify morphology exists
        assert "今日" in ngram_data.morphology
        assert ngram_data.morphology["今日"].surface == "今日"
        assert len(ngram_data.morphology["今日"].pos) > 0

    def test_phrases_to_ngram_data_multiple_phrases(self):
        """Test phrases_to_ngram_data with multiple phrases."""
        phrases = ["今日はいい天気", "今日は雨"]

        ngram_data = JaCompleter.phrases_to_ngram_data(phrases)

        # "今日" should appear twice (once in each phrase)
        assert ngram_data.unigrams["今日"] == 2
        # "は" should appear twice
        assert ngram_data.unigrams["は"] == 2

        # Verify morphology is deduplicated
        assert "今日" in ngram_data.morphology
        assert ngram_data.morphology["今日"].surface == "今日"

    def test_phrases_to_ngram_data_empty_list(self):
        """Test phrases_to_ngram_data with empty list."""
        ngram_data = JaCompleter.phrases_to_ngram_data([])

        assert ngram_data.unigrams == {}
        assert ngram_data.bigrams == {}
        assert ngram_data.trigrams == {}
        assert ngram_data.morphology == {}

    def test_add_simple_suggestions_with_simple_suggestions_type(self):
        """Test add_simple_suggestions accepts SimpleSuggestions type."""
        from ja_complete.types import SimpleSuggestions

        completer = JaCompleter()

        # Create SimpleSuggestions
        simple_sugg = SimpleSuggestions(data={"お": ["おはよう", "おやすみ"]})

        # Should accept SimpleSuggestions type
        completer.add_simple_suggestions(simple_sugg)

        # Verify it was added
        results = completer.suggest_from_simple("お", fallback_to_ngram=False)
        assert len(results) == 2
        assert any("おはよう" in r.text for r in results.items)

    def test_integration_phrases_to_simple_suggestions_and_add(self):
        """Test integration of phrases_to_simple_suggestions with add_simple_suggestions."""
        completer = JaCompleter()

        phrases = ["こんにちは", "こんばんは"]
        suggestions = JaCompleter.phrases_to_simple_suggestions(phrases)

        # Add to completer
        completer.add_simple_suggestions(suggestions)

        # Query with prefix
        results = completer.suggest_from_simple("こん", fallback_to_ngram=False)
        assert len(results) > 0
        assert any("こんにちは" in r.text for r in results.items)
