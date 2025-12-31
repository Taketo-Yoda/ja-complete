"""Pytestの設定と共有フィクスチャ。"""

import os

import pytest

from ja_complete.models.ngram import NgramModel


def pytest_configure(config):
    """高速テストのためにデフォルトモデルのロードをスキップするようpytestを設定。"""
    os.environ["SKIP_DEFAULT_MODEL"] = "1"


@pytest.fixture(scope="session")
def default_ngram_model():
    """
    テストセッションごとに1回だけデフォルトN-gramモデルを読み込む。

    このfixtureは7.7MBのデフォルトモデルをキャッシュし、
    各テストで再読み込みすることを避け、テスト性能を大幅に改善する。

    Returns:
        NgramModel: 読み込まれたデフォルトN-gramモデル
    """
    # デフォルトモデルを明示的に読み込む（skip_default=False）
    model = NgramModel(skip_default=False)
    return model


@pytest.fixture
def empty_ngram_model():
    """
    テスト用の空のN-gramモデルを作成する。

    テストがカスタムテストデータで動作する必要がある場合、
    大きなデフォルトモデルの読み込みを避ける。

    Returns:
        NgramModel: 空のN-gramモデル
    """
    # skip_default=Trueを使用してデフォルトモデルの読み込みを避ける
    model = NgramModel(skip_default=True)
    return model


@pytest.fixture
def sample_ngram_model():
    """
    サンプル日本語データを含む小さなN-gramモデルを作成する。

    Returns:
        NgramModel: 基本的な日本語テストデータを含むモデル
    """
    # skip_default=Trueを使用してデフォルトモデルの読み込みを避ける
    model = NgramModel(skip_default=True)
    model.unigrams = {"今日": 10, "は": 8, "晴れ": 5, "雨": 3}
    model.bigrams = {"今日": {"は": 8, "も": 2}}
    model.trigrams = {("今日", "は"): {"晴れ": 5, "雨": 3}}
    model.vocabulary_size = len(model.unigrams)
    return model
