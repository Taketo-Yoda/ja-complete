"""単純な辞書ベースの補完モデル。

このモデルは複雑な分析なしに、プレフィックスから候補への
直接マッピングを提供する。最もシンプルな補完戦略。
"""

from typing import Any

from ja_complete.models.base import CompletionModel


class SimpleDictModel(CompletionModel):
    """
    プレフィックスベース補完のための単純な辞書モデル。

    このモデルはプレフィックスを補完候補に直接マッピングする。
    一般的な挨拶、コマンドなどの固定語彙に有用。
    """

    def __init__(self) -> None:
        """空の候補辞書を初期化する。"""
        self.suggestions: dict[str, list[str]] = {}

    def add_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """
        プレフィックスマッピングを追加または更新する。

        Args:
            suggestions: プレフィックス -> 補完候補リストのマッピング辞書

        Example:
            >>> model = SimpleDictModel()
            >>> model.add_suggestions({
            ...     "お": ["おはよう", "おやすみ", "お疲れ様"],
            ...     "あり": ["ありがとう", "ありがとうございます"]
            ... })
        """
        self.suggestions.update(suggestions)

    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        単純辞書からプレフィックスマッチを返す。

        アルゴリズム:
        1. 候補辞書でinput_textを検索（完全一致）
        2. 完全一致がない場合、徐々に短いプレフィックスを試す
        3. score=1.0でマッチを返す
        4. top_k個まで結果を返す

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数

        Returns:
            'text'と'score'キーを持つ補完辞書のリスト

        Raises:
            ValueError: input_textが空、またはtop_k <= 0の場合
        """
        if not input_text:
            raise ValueError("input_text cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results: list[dict[str, Any]] = []

        # まず完全プレフィックスマッチを試す
        if input_text in self.suggestions:
            for text in self.suggestions[input_text][:top_k]:
                results.append({"text": text, "score": 1.0})
            return results

        # 徐々に短いプレフィックスを試す（フォールバック戦略）
        for length in range(len(input_text) - 1, 0, -1):
            prefix = input_text[:length]
            if prefix in self.suggestions:
                for text in self.suggestions[prefix][:top_k]:
                    # 部分プレフィックスマッチには低いスコア
                    score = length / len(input_text)
                    results.append({"text": text, "score": score})
                return results

        # マッチが見つからなかった
        return []
