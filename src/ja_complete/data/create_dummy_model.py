"""Create dummy N-gram model for initial development."""

import pickle
from pathlib import Path

# Dummy N-gram model with basic Japanese phrases
dummy_model = {
    "unigrams": {
        "今日": 10,
        "は": 20,
        "いい": 5,
        "天気": 5,
        "です": 15,
        "明日": 8,
        "雨": 4,
        "でしょう": 6,
        "ね": 12,
    },
    "bigrams": {
        "今日": {"は": 10},
        "は": {"いい": 5, "です": 8},
        "いい": {"天気": 5},
        "天気": {"です": 5},
        "です": {"ね": 10},
        "明日": {"は": 8},
        "雨": {"でしょう": 4},
    },
    "trigrams": {
        ("今日", "は"): {"いい": 5},
        ("は", "いい"): {"天気": 5},
        ("いい", "天気"): {"です": 5},
        ("明日", "は"): {"雨": 4},
        ("は", "雨"): {"でしょう": 3},
    },
}

# Save to package data directory
output_path = Path(__file__).parent / "default_ngram.pkl"
with open(output_path, "wb") as f:
    pickle.dump(dummy_model, f)

print(f"Dummy N-gram model created at: {output_path}")
