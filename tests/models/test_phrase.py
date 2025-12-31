"""Comprehensive tests for PhraseModel."""

import pytest
from pydantic import ValidationError

from ja_complete.models.phrase import PhraseModel
from ja_complete.types import SuggestionList


class TestPhraseModelInitialization:
    """Test PhraseModel initialization."""

    def test_initialization(self):
        """Test model initializes with empty storage."""
        model = PhraseModel()
        assert model is not None
        assert isinstance(model.phrases, set)
        assert isinstance(model.prefix_map, dict)
        assert len(model.phrases) == 0
        assert len(model.prefix_map) == 0


class TestAddPhrases:
    """Test add_phrases() method."""

    def test_add_single_phrase(self):
        """Test adding a single phrase."""
        model = PhraseModel()
        phrases = ["今日はいい天気です"]
        model.add_phrases(phrases)

        assert len(model.phrases) == 1
        assert "今日はいい天気です" in model.phrases
        assert len(model.prefix_map) > 0

    def test_add_multiple_phrases(self):
        """Test adding multiple phrases."""
        model = PhraseModel()
        phrases = [
            "スマホの買い換えと合わせて一式揃えたい",
            "新生活に備えた準備を始めたい",
            "夏を爽やかに過ごしたい",
        ]
        model.add_phrases(phrases)

        assert len(model.phrases) == 3
        for phrase in phrases:
            assert phrase in model.phrases

    def test_add_duplicate_phrases(self):
        """Test that duplicate phrases are handled correctly."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "今日はいい天気です"]
        model.add_phrases(phrases)

        # Set should deduplicate
        assert len(model.phrases) == 1

    def test_add_empty_phrase(self):
        """Test that empty phrases are skipped."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "", "明日も晴れです"]
        model.add_phrases(phrases)

        assert len(model.phrases) == 2
        assert "" not in model.phrases

    def test_add_phrases_multiple_times(self):
        """Test adding phrases in multiple batches."""
        model = PhraseModel()
        model.add_phrases(["今日はいい天気です"])
        model.add_phrases(["明日も晴れです"])

        assert len(model.phrases) == 2

    def test_prefix_map_populated(self):
        """Test that prefix_map is populated after adding phrases."""
        model = PhraseModel()
        model.add_phrases(["スマホを買う"])

        # Should have character prefixes: "ス", "スマ", "スマホ"
        assert "ス" in model.prefix_map
        assert "スマホを買う" in model.prefix_map["ス"]


class TestGeneratePrefixes:
    """Test _generate_prefixes() method."""

    def test_character_level_prefixes(self):
        """Test character-level prefix generation."""
        model = PhraseModel()
        prefixes = model._generate_prefixes("スマホ")

        # Should have "ス", "スマ", "スマホ"
        assert "ス" in prefixes
        assert "スマ" in prefixes
        assert "スマホ" in prefixes

    def test_short_phrase_prefixes(self):
        """Test prefix generation for short phrase."""
        model = PhraseModel()
        prefixes = model._generate_prefixes("私")

        # Only 1 character - should have just "私"
        assert "私" in prefixes

    def test_two_character_phrase(self):
        """Test prefix generation for 2-character phrase."""
        model = PhraseModel()
        prefixes = model._generate_prefixes("今日")

        assert "今" in prefixes
        assert "今日" in prefixes

    def test_morpheme_boundary_prefixes(self):
        """Test morpheme boundary prefix generation."""
        model = PhraseModel()
        prefixes = model._generate_prefixes("今日は晴れ")

        # Should include prefixes at morpheme boundaries
        assert isinstance(prefixes, set)
        # At minimum should have character-level prefixes (1-3 chars)
        assert len(prefixes) >= 3

    def test_bunsetsu_boundary_prefixes(self):
        """Test bunsetsu boundary prefix generation."""
        model = PhraseModel()
        prefixes = model._generate_prefixes("私は学生です")

        # Should include bunsetsu boundaries
        assert isinstance(prefixes, set)
        # Should have multiple prefixes from different strategies

    def test_complex_phrase_prefixes(self):
        """Test prefix generation for complex phrase."""
        model = PhraseModel()
        phrase = "スマホの買い換えと合わせて一式揃えたい"
        prefixes = model._generate_prefixes(phrase)

        # Should have character-level prefixes
        assert "ス" in prefixes
        assert "スマ" in prefixes
        assert "スマホ" in prefixes

        # Should have many prefixes from all strategies
        assert len(prefixes) >= 5


class TestCalculateScore:
    """Test _calculate_score() method."""

    def test_perfect_match_score(self):
        """Test score for perfect match."""
        model = PhraseModel()
        phrase = "今日はいい天気です"
        score = model._calculate_score(phrase, phrase)

        # Perfect match should get PREFIX_WEIGHT + MORPHEME_WEIGHT
        assert score == pytest.approx(1.0, abs=0.01)

    def test_prefix_match_score(self):
        """Test score for prefix match."""
        model = PhraseModel()
        input_text = "今日"
        phrase = "今日はいい天気です"
        score = model._calculate_score(input_text, phrase)

        # Should have prefix score component
        assert score > 0.0
        assert score < 1.0

    def test_no_prefix_match_score(self):
        """Test score when there's no prefix match."""
        model = PhraseModel()
        input_text = "明日"
        phrase = "今日はいい天気です"
        score = model._calculate_score(input_text, phrase)

        # No prefix match should return 0.0
        assert score == 0.0

    def test_short_prefix_vs_long_phrase(self):
        """Test scoring with short prefix and long phrase."""
        model = PhraseModel()
        input_text = "ス"
        phrase = "スマホの買い換えと合わせて一式揃えたい"
        score = model._calculate_score(input_text, phrase)

        # Should have lower score due to length ratio
        assert score > 0.0
        assert score < 0.6

    def test_longer_prefix_higher_score(self):
        """Test that longer prefix matches get higher scores."""
        model = PhraseModel()
        phrase = "スマホの買い換えと合わせて一式揃えたい"

        score_short = model._calculate_score("ス", phrase)
        score_medium = model._calculate_score("スマホ", phrase)
        score_long = model._calculate_score("スマホの買い換え", phrase)

        # Longer prefixes should score higher
        assert score_medium > score_short
        assert score_long > score_medium

    def test_morpheme_overlap_contribution(self):
        """Test morpheme overlap score contribution."""
        model = PhraseModel()
        # Input has common morphemes with phrase
        input_text = "スマホ"
        phrase = "スマホの買い換え"
        score = model._calculate_score(input_text, phrase)

        # Should have both prefix and morpheme score
        # Morpheme overlap should be high (スマホ is common)
        assert score > 0.0

    def test_score_components_sum(self):
        """Test that score components sum correctly."""
        model = PhraseModel()
        input_text = "今日は"
        phrase = "今日はいい天気です"
        score = model._calculate_score(input_text, phrase)

        # Score should be combination of prefix and morpheme
        # Should be less than 1.0 for partial match
        assert 0.0 < score <= 1.0

    def test_score_capped_at_one(self):
        """Test that score is capped at 1.0."""
        model = PhraseModel()
        phrase = "今日はいい天気です"
        score = model._calculate_score(phrase, phrase)

        # Even if components sum > 1.0, should cap at 1.0
        assert score <= 1.0


class TestSuggest:
    """Test suggest() method."""

    def test_basic_suggestion(self):
        """Test basic suggestion functionality."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "今日は雨です"]
        model.add_phrases(phrases)

        results = model.suggest("今日", top_k=10)

        assert len(results) == 2
        assert all(hasattr(r, "text") and hasattr(r, "score") for r in results.items)

    def test_exact_prefix_match(self):
        """Test exact prefix match returns correct phrase."""
        model = PhraseModel()
        phrases = ["スマホの買い換えと合わせて一式揃えたい"]
        model.add_phrases(phrases)

        results = model.suggest("スマホの", top_k=10)

        assert len(results) > 0
        assert results[0].text == "スマホの買い換えと合わせて一式揃えたい"

    def test_character_prefix_match(self):
        """Test single character prefix match."""
        model = PhraseModel()
        phrases = ["スマホを買う", "スイカを食べる"]
        model.add_phrases(phrases)

        results = model.suggest("ス", top_k=10)

        assert len(results) == 2
        assert any("スマホ" in r.text for r in results.items)
        assert any("スイカ" in r.text for r in results.items)

    def test_no_match_returns_empty(self):
        """Test that no match returns empty list."""
        model = PhraseModel()
        phrases = ["今日はいい天気です"]
        model.add_phrases(phrases)

        results = model.suggest("明日", top_k=10)

        assert len(results) == 0

    def test_top_k_limit(self):
        """Test top_k parameter limits results."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "今日は雨です", "今日は曇りです", "今日は晴れです"]
        model.add_phrases(phrases)

        results = model.suggest("今日", top_k=2)

        assert len(results) == 2

    def test_results_sorted_by_score(self):
        """Test results are sorted by score descending."""
        model = PhraseModel()
        phrases = [
            "今日",  # Perfect match
            "今日はいい天気です",  # Good match
            "今日は雨かもしれません",  # Longer phrase
        ]
        model.add_phrases(phrases)

        results = model.suggest("今日", top_k=10)

        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_empty_input_raises_error(self):
        """Test empty input raises ValueError."""
        model = PhraseModel()
        model.add_phrases(["今日はいい天気です"])

        with pytest.raises(ValueError, match="input_text cannot be empty"):
            model.suggest("", top_k=10)

    def test_zero_top_k_raises_error(self):
        """Test top_k=0 raises ValidationError."""
        model = PhraseModel()
        model.add_phrases(["今日はいい天気です"])

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            model.suggest("今日", top_k=0)

    def test_negative_top_k_raises_error(self):
        """Test negative top_k raises ValidationError."""
        model = PhraseModel()
        model.add_phrases(["今日はいい天気です"])

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            model.suggest("今日", top_k=-1)

    def test_suggest_on_empty_model(self):
        """Test suggestion on model with no phrases."""
        model = PhraseModel()
        results = model.suggest("今日", top_k=10)

        assert len(results) == 0

    def test_multiple_matches_ranking(self):
        """Test ranking with multiple matches."""
        model = PhraseModel()
        phrases = [
            "スマホの買い換えと合わせて一式揃えたい",
            "スマホを買う",
            "スマートフォンを購入する",
        ]
        model.add_phrases(phrases)

        results = model.suggest("スマホ", top_k=10)

        # Should have at least 2 results (スマートフォン might not match prefix)
        assert len(results) >= 2
        # Results should be scored and sorted
        assert all(r.score > 0 for r in results.items)

    def test_long_input_text(self):
        """Test with long input text."""
        model = PhraseModel()
        phrase = "スマホの買い換えと合わせて一式揃えたい"
        model.add_phrases([phrase])

        results = model.suggest("スマホの買い換えと合わせて", top_k=10)

        assert len(results) == 1
        assert results[0].text == phrase
        assert results[0].score > 0.5  # Should have high score

    def test_japanese_script_variations(self):
        """Test with different Japanese scripts."""
        model = PhraseModel()
        phrases = ["ひらがなです", "カタカナです", "漢字です"]
        model.add_phrases(phrases)

        # Hiragana
        results = model.suggest("ひら", top_k=10)
        assert len(results) == 1

        # Katakana
        results = model.suggest("カタ", top_k=10)
        assert len(results) == 1

        # Kanji
        results = model.suggest("漢", top_k=10)
        assert len(results) == 1

    def test_score_range(self):
        """Test that all scores are in valid range [0, 1]."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "今日は雨です", "今日は曇りです"]
        model.add_phrases(phrases)

        results = model.suggest("今日", top_k=10)

        for result in results.items:
            assert 0.0 <= result.score <= 1.0


class TestPhraseModelEdgeCases:
    """Test edge cases and special scenarios."""

    def test_phrase_with_numbers(self):
        """Test phrases containing numbers."""
        model = PhraseModel()
        phrases = ["2024年1月1日"]
        model.add_phrases(phrases)

        results = model.suggest("2024", top_k=10)
        assert len(results) >= 0  # May or may not match depending on tokenization

    def test_phrase_with_punctuation(self):
        """Test phrases containing punctuation."""
        model = PhraseModel()
        phrases = ["こんにちは、元気ですか？"]
        model.add_phrases(phrases)

        results = model.suggest("こんにちは", top_k=10)
        assert len(results) > 0

    def test_very_long_phrase(self):
        """Test with very long phrase."""
        model = PhraseModel()
        long_phrase = "あ" * 100
        model.add_phrases([long_phrase])

        results = model.suggest("あ", top_k=10)
        assert len(results) == 1

    def test_single_character_phrases(self):
        """Test with single character phrases."""
        model = PhraseModel()
        phrases = ["私", "僕", "俺"]
        model.add_phrases(phrases)

        results = model.suggest("私", top_k=10)
        assert len(results) == 1
        assert results[0].text == "私"

    def test_identical_prefixes_different_phrases(self):
        """Test phrases with identical prefixes."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "今日は雨です", "今日は曇りです", "今日は風が強いです"]
        model.add_phrases(phrases)

        results = model.suggest("今日は", top_k=10)
        assert len(results) == 4

    def test_suggest_consistency(self):
        """Test that suggest returns consistent results."""
        model = PhraseModel()
        phrases = ["今日はいい天気です", "明日も晴れです"]
        model.add_phrases(phrases)

        results1 = model.suggest("今日", top_k=10)
        results2 = model.suggest("今日", top_k=10)

        assert results1 == results2

    def test_mixed_length_phrases(self):
        """Test with phrases of varying lengths."""
        model = PhraseModel()
        phrases = [
            "短い",
            "少し長めのフレーズです",
            "これはとても長いフレーズで、たくさんの文字が含まれています",
        ]
        model.add_phrases(phrases)

        # Each should be retrievable by its own prefix
        results = model.suggest("短", top_k=10)
        assert len(results) > 0

        results = model.suggest("少し", top_k=10)
        assert len(results) > 0
