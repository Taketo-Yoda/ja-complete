"""N-gram統計補完モデル。

このモデルはbigramとtrigram統計を使用して、
ユーザー入力の可能性の高い継続を予測する。
"""

import pickle
import warnings
from pathlib import Path
from typing import Any

from ja_complete import tokenizer
from ja_complete.models.base import CompletionModel

# Laplaceスムージングパラメータ
SMOOTHING_ALPHA = 1.0


class NgramModel(CompletionModel):
    """
    N-gram統計補完モデル。

    bigramとtrigram確率を使用して補完を生成する。
    より良い確率推定のためLaplaceスムージングを実装。
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        N-gramモデルを初期化する。

        Args:
            model_path: pickleモデルファイルへのパス。
                       Noneの場合、パッケージデータからデフォルトモデルを読み込む。
        """
        self.unigrams: dict[str, int] = {}
        self.bigrams: dict[str, dict[str, int]] = {}
        self.trigrams: dict[tuple[str, str], dict[str, int]] = {}
        self.vocabulary_size: int = 0

        if model_path:
            self.load_model(model_path)
        else:
            self.load_default_model()

    def load_model(self, path: str) -> None:
        """
        ファイルからpickle化されたN-gramモデルを読み込む。

        Args:
            path: モデルファイルへのパス

        Raises:
            FileNotFoundError: モデルファイルが存在しない場合

        セキュリティ警告:
            このメソッドはpickle.load()を使用しており、任意のコードを実行できます。
            信頼できるソースからのモデルファイルのみを読み込んでください。
            信頼できない、または不明な出所のモデルを読み込まないでください。
        """
        model_file = Path(path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Warn about pickle security risk when loading non-default models
        default_model = Path(__file__).parent.parent / "data" / "default_ngram.pkl"
        if model_file.resolve() != default_model.resolve():
            warnings.warn(
                f"Loading model from {path}. "
                "WARNING: Pickle files can execute arbitrary code. "
                "Only load models from trusted sources.",
                RuntimeWarning,
                stacklevel=2,
            )

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Validate model structure
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model format: expected dict, got {type(model)}")

        # Warn if model is missing expected keys (but still allow loading)
        expected_keys = {"unigrams", "bigrams", "trigrams"}
        missing_keys = expected_keys - set(model.keys())
        if missing_keys:
            warnings.warn(
                f"Model is missing optional keys: {missing_keys}. "
                "These will be initialized as empty.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.unigrams = model.get("unigrams", {})
        self.bigrams = model.get("bigrams", {})
        self.trigrams = model.get("trigrams", {})
        self.vocabulary_size = len(self.unigrams)

    def load_default_model(self) -> None:
        """パッケージデータからデフォルトのN-gramモデルを読み込む。"""
        default_model_path = Path(__file__).parent.parent / "data" / "default_ngram.pkl"
        if default_model_path.exists():
            self.load_model(str(default_model_path))
        else:
            # デフォルトモデルが利用できない場合 - 空のモデルを使用
            self.vocabulary_size = 0

    def _calculate_probability(self, history: list[str], next_token: str) -> float:
        """
        N-gramを使用してhistoryが与えられたときのnext_tokenの確率を計算する。

        利用可能な場合はtrigramを使用し、bigramにフォールバック、その後unigramにフォールバック。
        Laplaceスムージングを適用。

        Args:
            history: コンテキストトークン（最後の1-2トークン）
            next_token: 確率を計算するトークン

        Returns:
            [0, 1]の確率スコア
        """
        if not self.vocabulary_size:
            return 0.0

        # trigram を試す（2個以上のhistoryトークンがある場合）
        if len(history) >= 2:
            trigram_key = (history[-2], history[-1])
            if trigram_key in self.trigrams:
                count = self.trigrams[trigram_key].get(next_token, 0)
                total = sum(self.trigrams[trigram_key].values())
                # Laplaceスムージング
                prob = (count + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * self.vocabulary_size)
                return prob

        # bigram を試す（1個以上のhistoryトークンがある場合）
        if len(history) >= 1:
            last_token = history[-1]
            if last_token in self.bigrams:
                count = self.bigrams[last_token].get(next_token, 0)
                total = sum(self.bigrams[last_token].values())
                # Laplaceスムージング
                prob = (count + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * self.vocabulary_size)
                return prob

        # unigramにフォールバック
        count = self.unigrams.get(next_token, 0)
        total = sum(self.unigrams.values())
        if total == 0:
            return 0.0
        prob = (count + SMOOTHING_ALPHA) / (total + SMOOTHING_ALPHA * self.vocabulary_size)
        return prob

    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        N-gram確率を使用して次の単語を予測する。

        アルゴリズム:
        1. input_textをトークン化
        2. 最後の1-2トークンをコンテキストとして取得
        3. 可能性のある全ての次のトークンの確率を計算
        4. 可能性の高い次のトークンを追加して補完を生成
        5. 確率でソートされたtop_k個の結果を返す

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数

        Returns:
            スコアの降順でソート済みの補完辞書のリスト

        Raises:
            ValueError: input_textが空、またはtop_k <= 0の場合
        """
        if not input_text:
            raise ValueError("input_text cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # 入力をトークン化
        tokens = tokenizer.tokenize(input_text)
        if not tokens:
            return []

        # コンテキストを取得（最後の1-2トークン）
        history = tokens[-2:] if len(tokens) >= 2 else tokens[-1:]

        # 候補となる次のトークンを取得
        candidates: dict[str, float] = {}

        # 2トークンのhistoryがある場合はtrigramを使用
        if len(history) == 2:
            trigram_key = (history[0], history[1])
            if trigram_key in self.trigrams:
                for next_token in self.trigrams[trigram_key]:
                    prob = self._calculate_probability(history, next_token)
                    candidates[next_token] = prob

        # 1個以上のhistoryトークンがある場合はbigramを使用
        if len(history) >= 1 and not candidates:
            last_token = history[-1]
            if last_token in self.bigrams:
                for next_token in self.bigrams[last_token]:
                    prob = self._calculate_probability(history, next_token)
                    candidates[next_token] = prob

        # まだ候補がない場合は全unigramを使用
        if not candidates and self.unigrams:
            for next_token in list(self.unigrams.keys())[:50]:  # 多すぎるのを避けるため制限
                prob = self._calculate_probability(history, next_token)
                candidates[next_token] = prob

        # 補完を構築
        results: list[dict[str, Any]] = []
        for next_token, prob in candidates.items():
            completion_text = input_text + next_token
            results.append({"text": completion_text, "score": prob})

        # 確率でソートしてtop_kを返す
        results.sort(key=lambda x: -x["score"])
        return results[:top_k]
