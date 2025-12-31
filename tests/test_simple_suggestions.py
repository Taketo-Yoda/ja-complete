"""Tests for SimpleSuggestions value class."""

import pytest
from pydantic import ValidationError

from ja_complete.types import SimpleSuggestions


class TestSimpleSuggestions:
    """Test SimpleSuggestions validation and behavior."""

    def test_valid_simple_suggestions(self):
        """Test creating valid SimpleSuggestions."""
        data = {
            "お": ["おはよう", "おやすみ"],
            "あり": ["ありがとう"],
        }
        suggestions = SimpleSuggestions(data=data)

        assert suggestions.data == data
        assert "お" in suggestions.data
        assert len(suggestions.data["お"]) == 2

    def test_empty_data_is_valid(self):
        """Test that empty data dict is valid."""
        suggestions = SimpleSuggestions(data={})
        assert suggestions.data == {}

    def test_default_factory(self):
        """Test default factory creates empty dict."""
        suggestions = SimpleSuggestions()
        assert suggestions.data == {}

    def test_to_dict_returns_copy(self):
        """Test to_dict() returns a copy."""
        data = {"お": ["おはよう"]}
        suggestions = SimpleSuggestions(data=data)

        result = suggestions.to_dict()
        assert result == data

        # Verify it's a copy (modifying result shouldn't affect original)
        result["new_key"] = ["test"]
        assert "new_key" not in suggestions.data

    def test_frozen_immutability(self):
        """Test that SimpleSuggestions is immutable (frozen)."""
        suggestions = SimpleSuggestions(data={"お": ["おはよう"]})

        with pytest.raises(ValidationError):
            suggestions.data = {"new": ["value"]}

    def test_empty_string_key_validation(self):
        """Test validation rejects empty string keys."""
        with pytest.raises(ValidationError, match="Empty string keys are not allowed"):
            SimpleSuggestions(data={"": ["value"]})

    def test_empty_list_validation(self):
        """Test validation rejects empty lists."""
        with pytest.raises(ValidationError, match="Empty list for key"):
            SimpleSuggestions(data={"key": []})

    def test_empty_string_in_values_validation(self):
        """Test validation rejects empty strings in values list."""
        with pytest.raises(ValidationError, match="Empty string in values list"):
            SimpleSuggestions(data={"お": ["おはよう", "", "おやすみ"]})

    def test_multiple_validation_errors(self):
        """Test that validation catches multiple errors."""
        # First error (empty list) should be caught
        with pytest.raises(ValidationError):
            SimpleSuggestions(data={
                "key1": [],
                "key2": ["value", ""],
            })
