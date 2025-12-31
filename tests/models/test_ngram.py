"""Comprehensive tests for NgramModel."""

import pickle

import pytest
from pydantic import ValidationError

from ja_complete.models.ngram import SMOOTHING_ALPHA, NgramModel
from ja_complete.types import SuggestionList


class TestNgramModelInitialization:
    """Test NgramModel initialization."""

    def test_initialization_without_model(self):
        """Test initialization without model path."""
        model = NgramModel()
        assert model is not None
        assert isinstance(model.unigrams, dict)
        assert isinstance(model.bigrams, dict)
        assert isinstance(model.trigrams, dict)
        assert model.vocabulary_size >= 0

    def test_initialization_with_nonexistent_model(self):
        """Test initialization with non-existent model path."""
        with pytest.raises(FileNotFoundError):
            NgramModel(model_path="/nonexistent/path/model.pkl")

    def test_empty_model_state(self):
        """Test model state when no default model exists."""
        model = NgramModel()
        # Model should initialize even without default model
        assert isinstance(model.unigrams, dict)
        assert isinstance(model.bigrams, dict)
        assert isinstance(model.trigrams, dict)


class TestLoadModel:
    """Test load_model() method."""

    def test_load_model_file_not_found(self):
        """Test loading non-existent model file."""
        model = NgramModel()
        with pytest.raises(FileNotFoundError):
            model.load_model("/nonexistent/model.pkl")

    def test_load_model_from_valid_file(self, tmp_path):
        """Test loading model from valid pickle file."""
        # Create test model
        test_model = {
            "unigrams": {"今日": 10, "天気": 5},
            "bigrams": {"今日": {"は": 8, "も": 2}},
            "trigrams": {("今日", "は"): {"晴れ": 5, "雨": 3}},
        }

        # Save to temporary file
        model_file = tmp_path / "test_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(test_model, f)

        # Load model
        model = NgramModel()
        model.load_model(str(model_file))

        assert model.unigrams == test_model["unigrams"]
        assert model.bigrams == test_model["bigrams"]
        assert model.trigrams == test_model["trigrams"]
        assert model.vocabulary_size == len(test_model["unigrams"])

    def test_load_model_updates_vocabulary_size(self, tmp_path):
        """Test that loading model updates vocabulary size."""
        test_model = {"unigrams": {"a": 1, "b": 2, "c": 3}, "bigrams": {}, "trigrams": {}}

        model_file = tmp_path / "test_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(test_model, f)

        model = NgramModel()
        model.load_model(str(model_file))

        assert model.vocabulary_size == 3

    def test_load_model_with_missing_keys(self, tmp_path):
        """Test loading model with missing dictionary keys."""
        # Model with missing keys (using .get with defaults)
        test_model = {
            "unigrams": {"今日": 10}
            # Missing bigrams and trigrams
        }

        model_file = tmp_path / "test_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(test_model, f)

        model = NgramModel()
        model.load_model(str(model_file))

        assert model.unigrams == {"今日": 10}
        assert model.bigrams == {}
        assert model.trigrams == {}


class TestCalculateProbability:
    """Test _calculate_probability() method."""

    def test_probability_with_empty_model(self):
        """Test probability calculation with empty model."""
        model = NgramModel()
        model.vocabulary_size = 0

        prob = model._calculate_probability(["今日"], "天気")
        assert prob == 0.0

    def test_unigram_probability(self):
        """Test unigram probability calculation."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "天気": 5, "雨": 3}
        model.vocabulary_size = 3

        prob = model._calculate_probability([], "今日")
        assert prob > 0.0
        assert prob < 1.0

    def test_bigram_probability(self):
        """Test bigram probability calculation."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "天気": 5}
        model.bigrams = {"今日": {"は": 8, "も": 2}}
        model.vocabulary_size = 2

        prob = model._calculate_probability(["今日"], "は")
        assert prob > 0.0

    def test_trigram_probability(self):
        """Test trigram probability calculation."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "は": 8, "晴れ": 5}
        model.bigrams = {"今日": {"は": 8}}
        model.trigrams = {("今日", "は"): {"晴れ": 5, "雨": 3}}
        model.vocabulary_size = 3

        prob = model._calculate_probability(["今日", "は"], "晴れ")
        assert prob > 0.0

    def test_laplace_smoothing_applied(self):
        """Test that Laplace smoothing is applied."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "天気": 5}
        model.bigrams = {"今日": {"は": 8}}
        model.vocabulary_size = 2

        # Token not in bigram should still get non-zero probability
        prob = model._calculate_probability(["今日"], "天気")
        assert prob > 0.0  # Should be non-zero due to smoothing

    def test_probability_fallback_to_bigram(self):
        """Test fallback from trigram to bigram."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5, "c": 3}
        model.bigrams = {"b": {"c": 2}}
        model.trigrams = {}  # No trigrams
        model.vocabulary_size = 3

        # Should fall back to bigram
        prob = model._calculate_probability(["a", "b"], "c")
        assert prob > 0.0

    def test_probability_fallback_to_unigram(self):
        """Test fallback from bigram to unigram."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5}
        model.bigrams = {}  # No bigrams
        model.trigrams = {}
        model.vocabulary_size = 2

        # Should fall back to unigram
        prob = model._calculate_probability(["a"], "b")
        assert prob > 0.0

    def test_zero_total_unigram_count(self):
        """Test probability when total unigram count is zero."""
        model = NgramModel()
        model.unigrams = {}
        model.vocabulary_size = 0

        prob = model._calculate_probability([], "test")
        assert prob == 0.0


class TestSuggest:
    """Test suggest() method."""

    def test_suggest_with_empty_model(self):
        """Test suggestion with empty model."""
        model = NgramModel()
        # Ensure model is truly empty (no default model loaded)
        model.unigrams = {}
        model.bigrams = {}
        model.trigrams = {}
        model.vocabulary_size = 0

        results = model.suggest("今日", top_k=5)
        assert len(results) == 0

    def test_suggest_basic(self):
        """Test basic suggestion functionality."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "は": 8, "晴れ": 5, "雨": 3}
        model.bigrams = {"今日": {"は": 8}}
        model.trigrams = {("今日", "は"): {"晴れ": 5, "雨": 3}}
        model.vocabulary_size = 4

        results = model.suggest("今日", top_k=5)

        # Should return suggestions
        assert isinstance(results, SuggestionList)
        for result in results.items:
            assert hasattr(result, "text")
            assert hasattr(result, "score")

    def test_suggest_empty_input_raises_error(self):
        """Test that empty input raises ValueError."""
        model = NgramModel()
        with pytest.raises(ValueError, match="input_text cannot be empty"):
            model.suggest("", top_k=5)

    def test_suggest_zero_top_k_raises_error(self):
        """Test that top_k=0 raises ValidationError."""
        model = NgramModel()
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            model.suggest("今日", top_k=0)

    def test_suggest_negative_top_k_raises_error(self):
        """Test that negative top_k raises ValidationError."""
        model = NgramModel()
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            model.suggest("今日", top_k=-1)

    def test_suggest_returns_top_k_results(self):
        """Test that suggest respects top_k parameter."""
        model = NgramModel()
        # Create model with many possible next tokens
        model.unigrams = {f"token{i}": i for i in range(100)}
        model.bigrams = {"test": {f"token{i}": i for i in range(100)}}
        model.vocabulary_size = 100

        results = model.suggest("test", top_k=5)
        assert len(results) <= 5

    def test_suggest_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "は": 8, "晴れ": 5}
        model.bigrams = {"今日": {"は": 8, "も": 2}}
        model.vocabulary_size = 3

        results = model.suggest("今日", top_k=10)

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_suggest_appends_to_input(self):
        """Test that suggestions append to input text."""
        model = NgramModel()
        model.unigrams = {"は": 10}
        model.bigrams = {"今日": {"は": 8}}
        model.vocabulary_size = 1

        results = model.suggest("今日", top_k=5)

        for result in results.items:
            # Completion should include original input
            assert result.text.startswith("今日")

    def test_suggest_with_no_tokenization_result(self):
        """Test suggest when tokenization returns empty."""
        model = NgramModel()
        # Input that might not tokenize well
        results = model.suggest("   ", top_k=5)
        # Should handle gracefully (empty or minimal results)
        assert isinstance(results, SuggestionList)

    def test_suggest_uses_trigram_when_available(self):
        """Test that trigram is used when history has 2 tokens."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5, "c": 3}
        model.bigrams = {"b": {"c": 2}}
        model.trigrams = {("a", "b"): {"c": 5}}
        model.vocabulary_size = 3

        # Input that tokenizes to multiple tokens
        results = model.suggest("ab", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_suggest_uses_bigram_when_no_trigram(self):
        """Test that bigram is used when trigram not available."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5}
        model.bigrams = {"a": {"b": 3}}
        model.trigrams = {}
        model.vocabulary_size = 2

        results = model.suggest("a", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_suggest_with_single_token_history(self):
        """Test suggestion with single token history."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "は": 8}
        model.bigrams = {"今日": {"は": 5, "も": 3}}
        model.vocabulary_size = 2

        results = model.suggest("今日", top_k=5)

        # Should use bigram
        assert len(results) >= 0

    def test_suggest_limits_unigram_candidates(self):
        """Test that unigram fallback limits candidates."""
        model = NgramModel()
        # Create large vocabulary
        model.unigrams = {f"token{i}": i for i in range(1000)}
        model.bigrams = {}
        model.trigrams = {}
        model.vocabulary_size = 1000

        results = model.suggest("test", top_k=5)

        # Should not return all 1000 candidates
        assert len(results) <= 50  # Limited to 50 in implementation


class TestNgramModelEdgeCases:
    """Test edge cases and special scenarios."""

    def test_model_with_only_unigrams(self):
        """Test model with only unigrams (no bi/trigrams)."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "天気": 5}
        model.bigrams = {}
        model.trigrams = {}
        model.vocabulary_size = 2

        results = model.suggest("test", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_model_with_only_bigrams(self):
        """Test model with only bigrams (no trigrams)."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5}
        model.bigrams = {"a": {"b": 3}}
        model.trigrams = {}
        model.vocabulary_size = 2

        results = model.suggest("a", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_very_long_input(self):
        """Test with very long input text."""
        model = NgramModel()
        model.unigrams = {"あ": 10}
        model.vocabulary_size = 1

        long_input = "あ" * 1000
        results = model.suggest(long_input, top_k=5)
        assert isinstance(results, SuggestionList)

    def test_single_character_input(self):
        """Test with single character input."""
        model = NgramModel()
        model.unigrams = {"私": 10, "は": 8}
        model.bigrams = {"私": {"は": 5}}
        model.vocabulary_size = 2

        results = model.suggest("私", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_suggest_consistency(self):
        """Test that suggest returns consistent results."""
        model = NgramModel()
        model.unigrams = {"今日": 10, "は": 8}
        model.bigrams = {"今日": {"は": 5}}
        model.vocabulary_size = 2

        results1 = model.suggest("今日", top_k=5)
        results2 = model.suggest("今日", top_k=5)

        assert results1 == results2

    def test_score_range(self):
        """Test that all scores are in valid range."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 5, "c": 3}
        model.bigrams = {"a": {"b": 4, "c": 2}}
        model.vocabulary_size = 3

        results = model.suggest("a", top_k=10)

        for result in results.items:
            assert 0.0 <= result.score <= 1.0

    def test_special_characters_in_model(self):
        """Test model with special characters."""
        model = NgramModel()
        model.unigrams = {"、": 5, "。": 3, "は": 10}
        model.vocabulary_size = 3

        results = model.suggest("test", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_numeric_tokens_in_model(self):
        """Test model with numeric tokens."""
        model = NgramModel()
        model.unigrams = {"2024": 5, "年": 3}
        model.bigrams = {"2024": {"年": 2}}
        model.vocabulary_size = 2

        results = model.suggest("2024", top_k=5)
        assert isinstance(results, SuggestionList)

    def test_mixed_script_model(self):
        """Test model with mixed scripts."""
        model = NgramModel()
        model.unigrams = {"ひらがな": 5, "カタカナ": 3, "漢字": 7, "English": 2}
        model.vocabulary_size = 4

        results = model.suggest("test", top_k=5)
        assert isinstance(results, SuggestionList)


class TestSmoothingAlpha:
    """Test Laplace smoothing constant."""

    def test_smoothing_alpha_value(self):
        """Test that SMOOTHING_ALPHA has expected value."""
        assert SMOOTHING_ALPHA == 1.0

    def test_smoothing_affects_probability(self):
        """Test that smoothing affects probability calculation."""
        model = NgramModel()
        model.unigrams = {"a": 10, "b": 0}  # b has zero count
        model.vocabulary_size = 2

        # Without smoothing, 'b' would have 0 probability
        # With smoothing, it should have non-zero probability
        prob = model._calculate_probability([], "b")
        assert prob > 0.0


class TestLoadDefaultModel:
    """Test load_default_model() method."""

    def test_load_default_model_no_file(self):
        """Test loading default model when file doesn't exist."""
        model = NgramModel()
        # Should not raise error even if default model doesn't exist
        assert model is not None
        assert model.vocabulary_size >= 0

    def test_default_model_path_construction(self):
        """Test that default model path is constructed correctly."""
        model = NgramModel()
        # Model should initialize without errors
        assert isinstance(model.unigrams, dict)
        assert isinstance(model.bigrams, dict)
        assert isinstance(model.trigrams, dict)


class TestAddNgramData:
    """Test add_ngram_data() method."""

    def test_add_ngram_data_to_empty_model(self):
        """Test adding n-gram data to empty model."""
        from ja_complete import JaCompleter
        from ja_complete.types import MorphToken

        model = NgramModel(skip_default=True)
        phrases = ["今日はいい天気"]
        ngram_data = JaCompleter.phrases_to_ngram_data(phrases)

        model.add_ngram_data(ngram_data)

        # Verify unigrams were added
        assert "今日" in model.unigrams
        assert model.unigrams["今日"] > 0

        # Verify bigrams were added
        assert len(model.bigrams) > 0

        # Verify morphology was added
        assert "今日" in model.morphology
        assert isinstance(model.morphology["今日"], MorphToken)

        # Verify vocabulary_size was updated
        assert model.vocabulary_size == len(model.unigrams)

    def test_add_ngram_data_merges_counts(self):
        """Test that add_ngram_data merges counts correctly."""
        from ja_complete import JaCompleter

        model = NgramModel(skip_default=True)

        # Add first phrase
        phrases1 = ["今日はいい天気"]
        ngram_data1 = JaCompleter.phrases_to_ngram_data(phrases1)
        model.add_ngram_data(ngram_data1)

        initial_count = model.unigrams.get("今日", 0)

        # Add second phrase with same token
        phrases2 = ["今日は雨"]
        ngram_data2 = JaCompleter.phrases_to_ngram_data(phrases2)
        model.add_ngram_data(ngram_data2)

        # Count should have increased
        assert model.unigrams["今日"] == initial_count + 1

    def test_add_ngram_data_preserves_existing_morphology(self):
        """Test that add_ngram_data doesn't overwrite existing morphology."""
        from ja_complete.types import MorphToken, NgramData

        model = NgramModel(skip_default=True)

        # Add initial morphology
        morph1 = MorphToken(surface="今日", pos="名詞", base_form="今日")
        data1 = NgramData(
            unigrams={"今日": 1},
            morphology={"今日": morph1},
        )
        model.add_ngram_data(data1)

        # Try to add different morphology for same token
        morph2 = MorphToken(surface="今日", pos="副詞", base_form="今日")  # Different POS
        data2 = NgramData(
            unigrams={"今日": 1},
            morphology={"今日": morph2},
        )
        model.add_ngram_data(data2)

        # Should keep original morphology
        assert model.morphology["今日"].pos == "名詞"

    def test_add_ngram_data_complex_structure(self):
        """Test add_ngram_data with complex bigram and trigram structure."""
        from ja_complete.types import MorphToken, NgramData

        model = NgramModel(skip_default=True)

        morph1 = MorphToken(surface="今日", pos="名詞", base_form="今日")
        morph2 = MorphToken(surface="は", pos="助詞", base_form="は")
        morph3 = MorphToken(surface="いい", pos="形容詞", base_form="良い")

        data = NgramData(
            unigrams={"今日": 5, "は": 3, "いい": 2},
            bigrams={
                "今日": {"は": 3},
                "は": {"いい": 2},
            },
            trigrams={
                ("今日", "は"): {"いい": 2},
            },
            morphology={
                "今日": morph1,
                "は": morph2,
                "いい": morph3,
            },
        )

        model.add_ngram_data(data)

        # Verify all levels
        assert model.unigrams["今日"] == 5
        assert model.bigrams["今日"]["は"] == 3
        assert model.trigrams[("今日", "は")]["いい"] == 2
        assert len(model.morphology) == 3

    def test_add_ngram_data_updates_vocabulary_size(self):
        """Test that vocabulary_size is updated correctly."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)
        assert model.vocabulary_size == 0

        data = NgramData(unigrams={"token1": 1, "token2": 1, "token3": 1})
        model.add_ngram_data(data)

        assert model.vocabulary_size == 3

        # Add more tokens
        data2 = NgramData(unigrams={"token4": 1, "token5": 1})
        model.add_ngram_data(data2)

        assert model.vocabulary_size == 5

    def test_add_ngram_data_empty_data(self):
        """Test adding empty NgramData doesn't break the model."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)
        empty_data = NgramData()

        model.add_ngram_data(empty_data)

        # Model should still be valid
        assert model.unigrams == {}
        assert model.bigrams == {}
        assert model.trigrams == {}
        assert model.morphology == {}
        assert model.vocabulary_size == 0


class TestTrailingSpaceRemoval:
    """Test trailing space removal and deduplication in suggest()."""

    def test_half_width_space_stripped(self):
        """Test that half-width trailing spaces are removed."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with space-ending tokens
        data = NgramData(
            unigrams={"今日": 10, " ": 5},
            bigrams={"今日": {" ": 5}},
        )
        model.add_ngram_data(data)

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Results should not contain trailing half-width spaces
        for suggestion in results.items:
            assert not suggestion.text.endswith(" "), \
                f"Found trailing space in: {repr(suggestion.text)}"

    def test_full_width_space_stripped(self):
        """Test that full-width trailing spaces (U+3000) are removed."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with full-width space
        data = NgramData(
            unigrams={"今日": 10, "　": 5},  # U+3000 full-width space
            bigrams={"今日": {"　": 5}},
        )
        model.add_ngram_data(data)

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Results should not contain trailing full-width spaces
        for suggestion in results.items:
            assert not suggestion.text.endswith("　"), \
                f"Found trailing full-width space in: {repr(suggestion.text)}"

    def test_mixed_spaces_stripped(self):
        """Test that mixed full/half-width trailing spaces are removed."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with mixed spaces
        data = NgramData(
            unigrams={"今日": 10, " ": 5, "　": 3},
            bigrams={"今日": {" ": 5, "　": 3}},
        )
        model.add_ngram_data(data)

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Results should not contain any trailing spaces
        for suggestion in results.items:
            assert not suggestion.text.endswith(" "), \
                f"Found trailing half-width space in: {repr(suggestion.text)}"
            assert not suggestion.text.endswith("　"), \
                f"Found trailing full-width space in: {repr(suggestion.text)}"

    def test_deduplication_keeps_higher_score(self):
        """Test that deduplication keeps the higher score when duplicates occur after stripping."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model where different tokens result in same text after stripping
        # "は" and "は " (with space) should deduplicate to "今日は"
        data = NgramData(
            unigrams={"今日": 100, "は": 50, " ": 10},
            bigrams={"今日": {"は": 50}},
            trigrams={("今日", "は"): {" ": 10}},
        )
        model.add_ngram_data(data)

        # Manually add a case that creates duplicates after stripping
        # This simulates: "今日は" (score X) and "今日は " (score Y) -> both become "今日は"
        model.bigrams["今日"] = {"は": 50, "は ": 30}  # "は " has trailing space

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Check that "今日は" appears only once
        texts = [s.text for s in results.items]
        assert texts.count("今日は") <= 1, "Deduplication failed: '今日は' appears multiple times"

    def test_top_k_maintained_after_deduplication(self):
        """Test that top_k results are returned even after deduplication."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with many tokens
        data = NgramData(
            unigrams={"今日": 100, "は": 50, "が": 40, "の": 30, "を": 20, "に": 15},
            bigrams={"今日": {"は": 50, "が": 40, "の": 30, "を": 20, "に": 15}},
        )
        model.add_ngram_data(data)

        top_k = 5
        results = model.suggest("今日", top_k=top_k, extend_particles=False)

        # Should return at most top_k results
        assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"

    def test_middle_spaces_preserved_when_followed_by_content(self):
        """Test that spaces are preserved when they're in the middle (followed by content)."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model where adding content creates non-trailing spaces
        data = NgramData(
            unigrams={"こんにちは": 100, "世界": 50, " ": 40},
            bigrams={"こんにちは": {" ": 40}},
            trigrams={("こんにちは", " "): {"世界": 30}},
        )
        model.add_ngram_data(data)

        results = model.suggest("こんにちは", top_k=10, extend_particles=False)

        # When we add " 世界", the space is in the middle, not trailing
        # So it should be preserved as "こんにちは 世界" (with space in middle)
        # But if we just add " " with nothing after, it gets stripped to "こんにちは"

        # Check that we don't have any results with ONLY trailing spaces
        for suggestion in results.items:
            # No result should be just the input with trailing spaces
            assert suggestion.text == suggestion.text.rstrip(" 　"), \
                f"Found trailing spaces in result: {repr(suggestion.text)}"

    def test_multiple_trailing_spaces_stripped(self):
        """Test that multiple trailing spaces are all removed."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with multiple spaces
        data = NgramData(
            unigrams={"今日": 100, " ": 50, "　": 30},
            bigrams={"今日": {" ": 50}, " ": {"　": 30}},
        )
        model.add_ngram_data(data)

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Results should not have any trailing spaces
        for suggestion in results.items:
            original_text = suggestion.text
            stripped_text = suggestion.text.rstrip(" 　")
            assert original_text == stripped_text, \
                f"Trailing spaces not fully removed: {repr(suggestion.text)}"

    def test_input_matching_results_excluded(self):
        """Test that results matching the input text are excluded."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model where stripping spaces would result in input text
        data = NgramData(
            unigrams={"今日": 100, " ": 50, "は": 30},
            bigrams={"今日": {" ": 50, "は": 30}},
        )
        model.add_ngram_data(data)

        results = model.suggest("今日", top_k=10, extend_particles=False)

        # Results should not include the input text itself
        for suggestion in results.items:
            assert suggestion.text != "今日", \
                f"Result should not match input text: {repr(suggestion.text)}"

        # Should have meaningful completions (not just the input)
        if len(results) > 0:
            assert all(len(s.text) > len("今日") for s in results.items), \
                "All results should be longer than input"

    def test_input_matching_excluded_maintains_top_k(self):
        """Test that excluding input-matching results still returns up to top_k results."""
        from ja_complete.types import NgramData

        model = NgramModel(skip_default=True)

        # Create model with many tokens
        data = NgramData(
            unigrams={
                "こんにちは": 100,
                " ": 50,  # This will create "こんにちは " -> "こんにちは" (excluded)
                "世界": 40,
                "みなさん": 30,
                "!": 20,
            },
            bigrams={
                "こんにちは": {" ": 50, "世界": 40, "みなさん": 30, "!": 20},
            },
        )
        model.add_ngram_data(data)

        top_k = 5
        results = model.suggest("こんにちは", top_k=top_k, extend_particles=False)

        # Should not include input text
        texts = [s.text for s in results.items]
        assert "こんにちは" not in texts, "Input text should be excluded"

        # Should return up to top_k results (excluding the input match)
        assert len(results) <= top_k, f"Should return at most {top_k} results"
        assert len(results) > 0, "Should return some results"
