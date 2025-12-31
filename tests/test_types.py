"""Comprehensive tests for types module (Suggestion and SuggestionList)."""

import pytest
from pydantic import ValidationError

from ja_complete.types import Suggestion, SuggestionList


class TestSuggestion:
    """Test Suggestion Value Object."""

    def test_create_valid_suggestion(self):
        """Test creating a valid suggestion."""
        suggestion = Suggestion(text="今日は晴れです", score=0.85)
        assert suggestion.text == "今日は晴れです"
        assert suggestion.score == 0.85

    def test_suggestion_with_min_score(self):
        """Test suggestion with minimum score (0.0)."""
        suggestion = Suggestion(text="test", score=0.0)
        assert suggestion.score == 0.0

    def test_suggestion_with_max_score(self):
        """Test suggestion with maximum score (1.0)."""
        suggestion = Suggestion(text="test", score=1.0)
        assert suggestion.score == 1.0

    def test_empty_text_raises_validation_error(self):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError, match="at least 1 character"):
            Suggestion(text="", score=0.5)

    def test_negative_score_raises_validation_error(self):
        """Test that negative score raises ValidationError."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Suggestion(text="test", score=-0.1)

    def test_score_above_one_raises_validation_error(self):
        """Test that score > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Suggestion(text="test", score=1.1)

    def test_suggestion_is_immutable(self):
        """Test that Suggestion is immutable (frozen=True)."""
        suggestion = Suggestion(text="test", score=0.5)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            suggestion.text = "modified"  # type: ignore
        with pytest.raises(ValidationError, match="Instance is frozen"):
            suggestion.score = 0.9  # type: ignore

    def test_suggestion_equality(self):
        """Test suggestion equality comparison."""
        s1 = Suggestion(text="test", score=0.5)
        s2 = Suggestion(text="test", score=0.5)
        s3 = Suggestion(text="test", score=0.6)
        s4 = Suggestion(text="other", score=0.5)

        assert s1 == s2
        assert s1 != s3
        assert s1 != s4


class TestSuggestionList:
    """Test SuggestionList Collection Object."""

    def test_create_empty_suggestion_list(self):
        """Test creating an empty suggestion list."""
        suggestion_list = SuggestionList(items=[])
        assert len(suggestion_list) == 0

    def test_create_with_suggestions(self):
        """Test creating suggestion list with items."""
        items = [
            Suggestion(text="今日は晴れ", score=0.8),
            Suggestion(text="今日は雨", score=0.6),
        ]
        suggestion_list = SuggestionList(items=items)
        assert len(suggestion_list) == 2

    def test_auto_sort_by_score_descending(self):
        """Test that items are automatically sorted by score descending."""
        items = [
            Suggestion(text="low", score=0.3),
            Suggestion(text="high", score=0.9),
            Suggestion(text="medium", score=0.6),
        ]
        suggestion_list = SuggestionList(items=items)

        # Should be sorted: high (0.9), medium (0.6), low (0.3)
        assert suggestion_list[0].score == 0.9
        assert suggestion_list[0].text == "high"
        assert suggestion_list[1].score == 0.6
        assert suggestion_list[1].text == "medium"
        assert suggestion_list[2].score == 0.3
        assert suggestion_list[2].text == "low"

    def test_top_k_method(self):
        """Test top_k method returns top k items."""
        items = [
            Suggestion(text=f"item{i}", score=i / 10)
            for i in range(10, 0, -1)  # scores 1.0, 0.9, ..., 0.1
        ]
        suggestion_list = SuggestionList(items=items)

        top_3 = suggestion_list.top_k(3)
        assert len(top_3) == 3
        assert top_3[0].score == 1.0
        assert top_3[1].score == 0.9
        assert top_3[2].score == 0.8

    def test_top_k_with_k_greater_than_length(self):
        """Test top_k when k > length returns all items."""
        items = [
            Suggestion(text="item1", score=0.8),
            Suggestion(text="item2", score=0.5),
        ]
        suggestion_list = SuggestionList(items=items)

        top_10 = suggestion_list.top_k(10)
        assert len(top_10) == 2

    def test_filter_by_score(self):
        """Test filter_by_score method."""
        items = [
            Suggestion(text="high1", score=0.9),
            Suggestion(text="high2", score=0.8),
            Suggestion(text="medium", score=0.6),
            Suggestion(text="low", score=0.3),
        ]
        suggestion_list = SuggestionList(items=items)

        filtered = suggestion_list.filter_by_score(0.7)
        assert len(filtered) == 2
        assert filtered[0].score == 0.9
        assert filtered[1].score == 0.8

    def test_filter_by_score_returns_new_instance(self):
        """Test that filter_by_score returns a new SuggestionList."""
        items = [Suggestion(text="test", score=0.5)]
        original = SuggestionList(items=items)
        filtered = original.filter_by_score(0.3)

        assert isinstance(filtered, SuggestionList)
        assert filtered is not original

    def test_to_dict_list(self):
        """Test to_dict_list converts to list of dicts."""
        items = [
            Suggestion(text="今日は晴れ", score=0.8),
            Suggestion(text="今日は雨", score=0.6),
        ]
        suggestion_list = SuggestionList(items=items)

        dict_list = suggestion_list.to_dict_list()
        assert isinstance(dict_list, list)
        assert len(dict_list) == 2
        assert dict_list[0] == {"text": "今日は晴れ", "score": 0.8}
        assert dict_list[1] == {"text": "今日は雨", "score": 0.6}

    def test_len_method(self):
        """Test __len__ returns correct length."""
        items = [Suggestion(text=f"item{i}", score=0.5) for i in range(5)]
        suggestion_list = SuggestionList(items=items)
        assert len(suggestion_list) == 5

    def test_getitem_method(self):
        """Test __getitem__ allows index access."""
        items = [
            Suggestion(text="first", score=0.9),
            Suggestion(text="second", score=0.5),
        ]
        suggestion_list = SuggestionList(items=items)

        assert suggestion_list[0].text == "first"
        assert suggestion_list[1].text == "second"

    def test_getitem_negative_index(self):
        """Test __getitem__ with negative index."""
        items = [
            Suggestion(text="first", score=0.9),
            Suggestion(text="second", score=0.5),
        ]
        suggestion_list = SuggestionList(items=items)

        assert suggestion_list[-1].text == "second"
        assert suggestion_list[-2].text == "first"

    def test_getitem_out_of_range(self):
        """Test __getitem__ raises IndexError for out of range."""
        suggestion_list = SuggestionList(items=[])
        with pytest.raises(IndexError):
            _ = suggestion_list[0]

    def test_iteration_over_items(self):
        """Test iteration over suggestion list items."""
        items = [
            Suggestion(text="item1", score=0.9),
            Suggestion(text="item2", score=0.5),
        ]
        suggestion_list = SuggestionList(items=items)

        # Iterate over .items
        collected = []
        for item in suggestion_list.items:
            collected.append(item.text)

        assert collected == ["item1", "item2"]

    def test_empty_list_operations(self):
        """Test operations on empty list."""
        suggestion_list = SuggestionList(items=[])

        assert len(suggestion_list) == 0
        assert suggestion_list.top_k(5) == []
        assert len(suggestion_list.filter_by_score(0.5)) == 0
        assert suggestion_list.to_dict_list() == []

    def test_sort_stability_with_equal_scores(self):
        """Test sort stability when scores are equal."""
        items = [
            Suggestion(text="first", score=0.5),
            Suggestion(text="second", score=0.5),
            Suggestion(text="third", score=0.5),
        ]
        suggestion_list = SuggestionList(items=items)

        # Python's sort is stable, so original order should be preserved
        # for items with equal scores
        assert suggestion_list[0].text == "first"
        assert suggestion_list[1].text == "second"
        assert suggestion_list[2].text == "third"

    def test_filter_by_score_with_exact_match(self):
        """Test filter_by_score includes items with exact min_score."""
        items = [
            Suggestion(text="exact", score=0.5),
            Suggestion(text="above", score=0.6),
            Suggestion(text="below", score=0.4),
        ]
        suggestion_list = SuggestionList(items=items)

        filtered = suggestion_list.filter_by_score(0.5)
        assert len(filtered) == 2
        assert filtered[0].text == "above"
        assert filtered[1].text == "exact"

    def test_large_suggestion_list(self):
        """Test with large number of suggestions."""
        items = [
            Suggestion(text=f"item{i}", score=i / 1000)
            for i in range(1000, 0, -1)
        ]
        suggestion_list = SuggestionList(items=items)

        assert len(suggestion_list) == 1000
        assert suggestion_list[0].score == 1.0
        assert suggestion_list[-1].score == 0.001

        top_10 = suggestion_list.top_k(10)
        assert len(top_10) == 10
        assert top_10[0].score == 1.0
        assert top_10[9].score == 0.991
