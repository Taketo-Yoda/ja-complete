"""単純な辞書ベースの補完モデル。

このモデルは複雑な分析なしに、プレフィックスから候補への
直接マッピングを提供する。最もシンプルな補完戦略。
"""

from pydantic import validate_call

from ja_complete.models.base import CompletionModel
from ja_complete.types import SimpleSuggestions, Suggestion, SuggestionList, TopK


class SimpleDictModel(CompletionModel):
    """
    プレフィックスベース補完のための単純な辞書モデル。

    このモデルはプレフィックスを補完候補に直接マッピングする。
    一般的な挨拶、コマンドなどの固定語彙に有用。
    """

    def __init__(self) -> None:
        """空の候補辞書を初期化する。"""
        self.suggestions: dict[str, list[str]] = {}

    def add_suggestions(self, suggestions: dict[str, list[str]] | SimpleSuggestions) -> None:
        """プレフィックスマッピングを追加または更新する。

        Args:
            suggestions: プレフィックス -> 補完候補リストのマッピング辞書、
                        またはSimpleSuggestions値オブジェクト

        Example:
            >>> model = SimpleDictModel()
            >>> # dict形式
            >>> model.add_suggestions({
            ...     "お": ["おはよう", "おやすみ", "お疲れ様"],
            ...     "あり": ["ありがとう", "ありがとうございます"]
            ... })
            >>> # SimpleSuggestions形式
            >>> from ja_complete.types import SimpleSuggestions
            >>> simple_sugg = SimpleSuggestions(data={"こ": ["こんにちは"]})
            >>> model.add_suggestions(simple_sugg)
        """
        if isinstance(suggestions, SimpleSuggestions):
            self.suggestions.update(suggestions.to_dict())
        else:
            self.suggestions.update(suggestions)

    @validate_call
    def suggest(self, input_text: str, top_k: TopK = 10) -> SuggestionList:
        """
        単純辞書からプレフィックスマッチを返す。

        アルゴリズム:
        1. 候補辞書でinput_textを検索（完全一致）
        2. 完全一致がない場合、徐々に短いプレフィックスを試す
        3. score=1.0でマッチを返す
        4. top_k個まで結果を返す

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数（1〜1000）

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        Raises:
            ValidationError: top_kが1〜1000の範囲外の場合
            ValueError: input_textが空の場合
        """
        if not input_text:
            raise ValueError("input_text cannot be empty")

        suggestions: list[Suggestion] = []

        # まず完全プレフィックスマッチを試す
        if input_text in self.suggestions:
            for text in self.suggestions[input_text]:
                if text:  # Skip empty strings
                    suggestions.append(Suggestion(text=text, score=1.0))
            suggestion_list = SuggestionList(items=suggestions)
            return SuggestionList(items=suggestion_list.top_k(top_k))

        # 徐々に短いプレフィックスを試す（フォールバック戦略）
        for length in range(len(input_text) - 1, 0, -1):
            prefix = input_text[:length]
            if prefix in self.suggestions:
                for text in self.suggestions[prefix]:
                    if text:  # Skip empty strings
                        # 部分プレフィックスマッチには低いスコア
                        score = length / len(input_text)
                        suggestions.append(Suggestion(text=text, score=score))
                suggestion_list = SuggestionList(items=suggestions)
                return SuggestionList(items=suggestion_list.top_k(top_k))

        # マッチが見つからなかった
        return SuggestionList(items=[])
