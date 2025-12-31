"""Tests for MorphToken and NgramData value classes."""

import pytest
from pydantic import ValidationError

from ja_complete.types import MorphToken, NgramData


class TestMorphToken:
    """Test MorphToken validation and behavior."""

    def test_valid_morph_token(self):
        """Test creating valid MorphToken."""
        token = MorphToken(surface="今日", pos="名詞", base_form="今日")

        assert token.surface == "今日"
        assert token.pos == "名詞"
        assert token.base_form == "今日"

    def test_frozen_immutability(self):
        """Test that MorphToken is immutable (frozen)."""
        token = MorphToken(surface="今日", pos="名詞", base_form="今日")

        with pytest.raises(ValidationError):
            token.surface = "明日"

    def test_empty_surface_validation(self):
        """Test validation rejects empty surface."""
        with pytest.raises(ValidationError):
            MorphToken(surface="", pos="名詞", base_form="今日")

    def test_empty_pos_validation(self):
        """Test validation rejects empty pos."""
        with pytest.raises(ValidationError):
            MorphToken(surface="今日", pos="", base_form="今日")

    def test_empty_base_form_validation(self):
        """Test validation rejects empty base_form."""
        with pytest.raises(ValidationError):
            MorphToken(surface="今日", pos="名詞", base_form="")


class TestNgramData:
    """Test NgramData validation and behavior."""

    def test_valid_ngram_data(self):
        """Test creating valid NgramData."""
        morph = MorphToken(surface="今日", pos="名詞", base_form="今日")
        data = NgramData(
            unigrams={"今日": 10, "は": 8},
            bigrams={"今日": {"は": 8}},
            trigrams={("今日", "は"): {"いい": 5}},
            morphology={"今日": morph},
        )

        assert data.unigrams["今日"] == 10
        assert data.bigrams["今日"]["は"] == 8
        assert data.trigrams[("今日", "は")]["いい"] == 5
        assert data.morphology["今日"].surface == "今日"

    def test_empty_data_is_valid(self):
        """Test that empty data is valid."""
        data = NgramData()

        assert data.unigrams == {}
        assert data.bigrams == {}
        assert data.trigrams == {}
        assert data.morphology == {}

    def test_default_factory(self):
        """Test default factory creates empty dicts."""
        data = NgramData()

        assert isinstance(data.unigrams, dict)
        assert isinstance(data.bigrams, dict)
        assert isinstance(data.trigrams, dict)
        assert isinstance(data.morphology, dict)

    def test_unigrams_positive_count_validation(self):
        """Test validation rejects non-positive unigram counts."""
        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(unigrams={"今日": 0})

        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(unigrams={"今日": -1})

    def test_bigrams_positive_count_validation(self):
        """Test validation rejects non-positive bigram counts."""
        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(bigrams={"今日": {"は": 0}})

        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(bigrams={"今日": {"は": -5}})

    def test_trigrams_positive_count_validation(self):
        """Test validation rejects non-positive trigram counts."""
        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(trigrams={("今日", "は"): {"いい": 0}})

        with pytest.raises(ValidationError, match="Count must be positive"):
            NgramData(trigrams={("今日", "は"): {"いい": -2}})

    def test_complex_ngram_structure(self):
        """Test NgramData with complex nested structure."""
        data = NgramData(
            unigrams={"今日": 10, "は": 15, "いい": 8, "天気": 5},
            bigrams={
                "今日": {"は": 10},
                "は": {"いい": 8},
                "いい": {"天気": 5},
            },
            trigrams={
                ("今日", "は"): {"いい": 8},
                ("は", "いい"): {"天気": 5},
            },
        )

        # Verify structure
        assert len(data.unigrams) == 4
        assert len(data.bigrams) == 3
        assert len(data.trigrams) == 2
        assert data.bigrams["今日"]["は"] == 10
        assert data.trigrams[("今日", "は")]["いい"] == 8

    def test_morphology_with_multiple_tokens(self):
        """Test morphology field with multiple tokens."""
        morph1 = MorphToken(surface="今日", pos="名詞", base_form="今日")
        morph2 = MorphToken(surface="は", pos="助詞", base_form="は")

        data = NgramData(
            morphology={
                "今日": morph1,
                "は": morph2,
            }
        )

        assert len(data.morphology) == 2
        assert data.morphology["今日"].pos == "名詞"
        assert data.morphology["は"].pos == "助詞"
