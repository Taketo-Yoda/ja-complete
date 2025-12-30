"""補完モデルの基底クラス。

このモジュールは、全ての補完モデルが実装しなければならない
抽象インターフェースを定義する。
交換可能な補完戦略のためにStrategyパターンに従う。
"""

from abc import ABC, abstractmethod
from typing import Any


class CompletionModel(ABC):
    """
    全ての補完モデルの抽象基底クラス。

    異なる補完戦略（フレーズベース、N-gram、単純辞書）間で
    一貫したインターフェースを強制し、StrategyパターンとOpen/Closed原則に従う。
    """

    @abstractmethod
    def suggest(self, input_text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        入力テキストの補完候補を生成する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 返す候補の最大数

        Returns:
            以下のキーを持つ辞書のリスト:
                - 'text' (str): 補完テキスト
                - 'score' (float): [0, 1]の範囲のスコア、高いほど良い
            スコアの降順でソート済み

        Raises:
            ValueError: input_textが空、またはtop_k <= 0の場合
        """
        pass
