"""Pytest configuration and shared fixtures."""

import os
import pickle
from pathlib import Path

import pytest

from ja_complete.models.ngram import NgramModel


def pytest_configure(config):
    """Configure pytest to skip default model loading for faster tests."""
    os.environ["SKIP_DEFAULT_MODEL"] = "1"


@pytest.fixture(scope="session")
def default_ngram_model():
    """
    Load the default N-gram model once per test session.

    This fixture caches the 591MB default model to avoid reloading
    it for every test, significantly improving test performance.

    Returns:
        NgramModel: The loaded default N-gram model
    """
    # Explicitly load default model (skip_default=False)
    model = NgramModel(skip_default=False)
    return model


@pytest.fixture
def empty_ngram_model():
    """
    Create an empty N-gram model for testing.

    This avoids loading the large default model when tests
    need to work with custom test data.

    Returns:
        NgramModel: An empty N-gram model
    """
    # Use skip_default=True to avoid loading default model
    model = NgramModel(skip_default=True)
    return model


@pytest.fixture
def sample_ngram_model():
    """
    Create a small N-gram model with sample Japanese data.

    Returns:
        NgramModel: A model with basic Japanese test data
    """
    # Use skip_default=True to avoid loading default model
    model = NgramModel(skip_default=True)
    model.unigrams = {"今日": 10, "は": 8, "晴れ": 5, "雨": 3}
    model.bigrams = {"今日": {"は": 8, "も": 2}}
    model.trigrams = {("今日", "は"): {"晴れ": 5, "雨": 3}}
    model.vocabulary_size = len(model.unigrams)
    return model
