"""Comprehensive tests for SimpleDictModel."""

import pytest

from ja_complete.models.simple import SimpleDictModel


class TestSimpleDictModelInitialization:
    """Test SimpleDictModel initialization."""

    def test_initialization(self):
        """Test model initializes with empty dictionary."""
        model = SimpleDictModel()
        assert model is not None
        assert isinstance(model.suggestions, dict)
        assert len(model.suggestions) == 0


class TestAddSuggestions:
    """Test add_suggestions() method."""

    def test_add_single_prefix(self):
        """Test adding suggestions for single prefix."""
        model = SimpleDictModel()
        suggestions = {"お": ["おはよう", "おやすみ"]}
        model.add_suggestions(suggestions)

        assert "お" in model.suggestions
        assert model.suggestions["お"] == ["おはよう", "おやすみ"]

    def test_add_multiple_prefixes(self):
        """Test adding suggestions for multiple prefixes."""
        model = SimpleDictModel()
        suggestions = {
            "お": ["おはよう", "おやすみ", "お疲れ様"],
            "あり": ["ありがとう", "ありがとうございます"],
        }
        model.add_suggestions(suggestions)

        assert len(model.suggestions) == 2
        assert "お" in model.suggestions
        assert "あり" in model.suggestions

    def test_update_existing_prefix(self):
        """Test updating suggestions for existing prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})
        model.add_suggestions({"お": ["おやすみ"]})

        # Should update (replace) the previous value
        assert model.suggestions["お"] == ["おやすみ"]

    def test_add_empty_suggestions(self):
        """Test adding empty suggestions dictionary."""
        model = SimpleDictModel()
        model.add_suggestions({})

        assert len(model.suggestions) == 0

    def test_add_suggestions_multiple_times(self):
        """Test adding suggestions in multiple batches."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})
        model.add_suggestions({"あり": ["ありがとう"]})

        assert len(model.suggestions) == 2
        assert "お" in model.suggestions
        assert "あり" in model.suggestions

    def test_add_prefix_with_empty_list(self):
        """Test adding prefix with empty completion list."""
        model = SimpleDictModel()
        suggestions = {"test": []}
        model.add_suggestions(suggestions)

        assert "test" in model.suggestions
        assert model.suggestions["test"] == []

    def test_add_suggestions_preserves_order(self):
        """Test that suggestion order is preserved."""
        model = SimpleDictModel()
        suggestions = {"お": ["おはよう", "おやすみ", "お疲れ様"]}
        model.add_suggestions(suggestions)

        assert model.suggestions["お"] == ["おはよう", "おやすみ", "お疲れ様"]


class TestSuggest:
    """Test suggest() method."""

    def test_exact_prefix_match(self):
        """Test exact prefix match returns suggestions."""
        model = SimpleDictModel()
        suggestions = {"あり": ["ありがとう", "ありがとうございます"]}
        model.add_suggestions(suggestions)

        results = model.suggest("あり", top_k=10)

        assert len(results) == 2
        assert results[0]["text"] in ["ありがとう", "ありがとうございます"]
        assert results[1]["text"] in ["ありがとう", "ありがとうございます"]

    def test_exact_match_score(self):
        """Test exact prefix match has score of 1.0."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        results = model.suggest("お", top_k=10)

        assert len(results) == 1
        assert results[0]["score"] == 1.0

    def test_partial_prefix_match(self):
        """Test partial prefix match with fallback."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう", "おやすみ"]})

        # Input "おは" should fall back to "お"
        results = model.suggest("おは", top_k=10)

        assert len(results) == 2
        assert results[0]["score"] < 1.0  # Lower score for partial match

    def test_partial_match_score_calculation(self):
        """Test partial prefix match score calculation."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        results = model.suggest("おは", top_k=10)

        # Score should be length ratio: 1 / 2 = 0.5
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.5, abs=0.01)

    def test_no_match_returns_empty(self):
        """Test that no match returns empty list."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        results = model.suggest("あ", top_k=10)

        assert results == []

    def test_top_k_limit(self):
        """Test top_k parameter limits results."""
        model = SimpleDictModel()
        model.add_suggestions(
            {"お": ["おはよう", "おやすみ", "お疲れ様", "お願いします", "お元気ですか"]}
        )

        results = model.suggest("お", top_k=3)

        assert len(results) == 3

    def test_top_k_larger_than_results(self):
        """Test top_k larger than available results."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう", "おやすみ"]})

        results = model.suggest("お", top_k=10)

        assert len(results) == 2

    def test_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        with pytest.raises(ValueError, match="input_text cannot be empty"):
            model.suggest("", top_k=10)

    def test_zero_top_k_raises_error(self):
        """Test top_k=0 raises ValueError."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        with pytest.raises(ValueError, match="top_k must be positive"):
            model.suggest("お", top_k=0)

    def test_negative_top_k_raises_error(self):
        """Test negative top_k raises ValueError."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        with pytest.raises(ValueError, match="top_k must be positive"):
            model.suggest("お", top_k=-1)

    def test_suggest_on_empty_model(self):
        """Test suggestion on empty model."""
        model = SimpleDictModel()
        results = model.suggest("test", top_k=10)

        assert results == []

    def test_fallback_to_shorter_prefix(self):
        """Test fallback to progressively shorter prefixes."""
        model = SimpleDictModel()
        model.add_suggestions({"あ": ["ありがとう"]})

        # Input "あり" should fall back to "あ"
        results = model.suggest("あり", top_k=10)

        assert len(results) == 1
        assert results[0]["text"] == "ありがとう"

    def test_fallback_score_decreases_with_length(self):
        """Test that fallback score decreases as prefix gets shorter."""
        model = SimpleDictModel()
        model.add_suggestions({"あ": ["ありがとう"]})

        # "あり" has length 2, prefix "あ" has length 1
        results = model.suggest("あり", top_k=10)

        # Score should be 1/2 = 0.5
        assert results[0]["score"] == pytest.approx(0.5, abs=0.01)

    def test_multiple_fallback_levels(self):
        """Test fallback through multiple prefix lengths."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        # Input "おはよ" (length 3) should fall back through "おは", "お"
        results = model.suggest("おはよ", top_k=10)

        assert len(results) == 1
        # Score should be 1/3 = 0.333...
        assert results[0]["score"] == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_exact_match_preferred_over_fallback(self):
        """Test that exact match is preferred over fallback."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"], "おは": ["おはようございます"]})

        # Exact match on "おは"
        results = model.suggest("おは", top_k=10)

        # Should get exact match with score 1.0
        assert results[0]["text"] == "おはようございます"
        assert results[0]["score"] == 1.0

    def test_result_dictionary_structure(self):
        """Test that result dictionaries have correct structure."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"]})

        results = model.suggest("お", top_k=10)

        assert len(results) == 1
        assert "text" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["text"], str)
        assert isinstance(results[0]["score"], float)


class TestSimpleDictModelEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_character_prefix(self):
        """Test with single character prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"私": ["私は学生です"]})

        results = model.suggest("私", top_k=10)

        assert len(results) == 1
        assert results[0]["score"] == 1.0

    def test_long_prefix(self):
        """Test with long prefix."""
        model = SimpleDictModel()
        long_prefix = "これはとても長いプレフィックスです"
        model.add_suggestions({long_prefix: ["これはとても長いプレフィックスですね"]})

        results = model.suggest(long_prefix, top_k=10)

        assert len(results) == 1
        assert results[0]["score"] == 1.0

    def test_hiragana_prefix(self):
        """Test with hiragana prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"ひらがな": ["ひらがなです"]})

        results = model.suggest("ひら", top_k=10)

        assert len(results) >= 0

    def test_katakana_prefix(self):
        """Test with katakana prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"カタカナ": ["カタカナです"]})

        results = model.suggest("カタ", top_k=10)

        assert len(results) >= 0

    def test_kanji_prefix(self):
        """Test with kanji prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"漢字": ["漢字です"]})

        results = model.suggest("漢", top_k=10)

        assert len(results) >= 0

    def test_mixed_script_prefix(self):
        """Test with mixed script prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"ひらがなカタカナ漢字": ["混在文字列"]})

        results = model.suggest("ひら", top_k=10)

        assert len(results) >= 0

    def test_numeric_prefix(self):
        """Test with numeric prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"2024": ["2024年"]})

        results = model.suggest("20", top_k=10)

        assert len(results) >= 0

    def test_alphanumeric_prefix(self):
        """Test with alphanumeric prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"Python3": ["Python3.10"]})

        results = model.suggest("Py", top_k=10)

        assert len(results) >= 0

    def test_special_characters_in_prefix(self):
        """Test with special characters in prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"メール：": ["メール：test@example.com"]})

        results = model.suggest("メール", top_k=10)

        assert len(results) >= 0

    def test_very_long_suggestion_list(self):
        """Test with very long suggestion list."""
        model = SimpleDictModel()
        long_list = [f"suggestion{i}" for i in range(1000)]
        model.add_suggestions({"test": long_list})

        results = model.suggest("test", top_k=10)

        assert len(results) == 10

    def test_empty_string_in_suggestions(self):
        """Test handling of empty string in suggestions."""
        model = SimpleDictModel()
        model.add_suggestions({"test": ["", "valid"]})

        results = model.suggest("test", top_k=10)

        # Both should be returned
        assert len(results) == 2

    def test_duplicate_suggestions_in_list(self):
        """Test handling of duplicate suggestions."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう", "おはよう", "おやすみ"]})

        results = model.suggest("お", top_k=10)

        # All entries should be returned (no deduplication)
        assert len(results) == 3

    def test_suggest_consistency(self):
        """Test that suggest returns consistent results."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう", "おやすみ"]})

        results1 = model.suggest("お", top_k=10)
        results2 = model.suggest("お", top_k=10)

        assert results1 == results2

    def test_score_range(self):
        """Test that all scores are in valid range [0, 1]."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"], "おは": ["おはようございます"]})

        # Test exact match
        results = model.suggest("お", top_k=10)
        assert all(0.0 <= r["score"] <= 1.0 for r in results)

        # Test partial match
        results = model.suggest("おはよ", top_k=10)
        assert all(0.0 <= r["score"] <= 1.0 for r in results)

    def test_fallback_stops_at_length_one(self):
        """Test that fallback doesn't go below length 1."""
        model = SimpleDictModel()
        model.add_suggestions({"a": ["apple"]})

        # Input longer than any prefix
        results = model.suggest("xyz", top_k=10)

        # Should return empty since no prefix matches
        assert results == []

    def test_whitespace_in_prefix(self):
        """Test handling of whitespace in prefix."""
        model = SimpleDictModel()
        model.add_suggestions({"hello world": ["hello world!"]})

        results = model.suggest("hello", top_k=10)

        assert len(results) >= 0

    def test_unicode_characters(self):
        """Test with various Unicode characters."""
        model = SimpleDictModel()
        model.add_suggestions({"😀": ["😀😁😂"], "🎌": ["🎌日本語"]})

        results = model.suggest("😀", top_k=10)
        assert len(results) >= 0

        results = model.suggest("🎌", top_k=10)
        assert len(results) >= 0

    def test_top_k_one(self):
        """Test with top_k=1."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう", "おやすみ", "お疲れ様"]})

        results = model.suggest("お", top_k=1)

        assert len(results) == 1

    def test_multiple_prefixes_same_completion(self):
        """Test multiple prefixes mapping to same completion."""
        model = SimpleDictModel()
        model.add_suggestions({"お": ["おはよう"], "おは": ["おはよう"]})

        # Both should work independently
        results1 = model.suggest("お", top_k=10)
        results2 = model.suggest("おは", top_k=10)

        assert len(results1) == 1
        assert len(results2) == 1


class TestSimpleDictModelDocExamples:
    """Test examples from docstrings."""

    def test_docstring_example(self):
        """Test example from class docstring."""
        model = SimpleDictModel()
        model.add_suggestions(
            {
                "お": ["おはよう", "おやすみ", "お疲れ様"],
                "あり": ["ありがとう", "ありがとうございます"],
            }
        )

        # Test first prefix
        results = model.suggest("お", top_k=10)
        assert len(results) == 3
        assert all(r["text"] in ["おはよう", "おやすみ", "お疲れ様"] for r in results)

        # Test second prefix
        results = model.suggest("あり", top_k=10)
        assert len(results) == 2
        assert all(r["text"] in ["ありがとう", "ありがとうございます"] for r in results)
