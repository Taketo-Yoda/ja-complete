"""カスタム型定義。

このモジュールは、ドメイン固有の型とバリデーションを提供する。
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field

# top_kの有効範囲: 1〜1000
# 1未満または1000を超える値は無効
TopK = Annotated[
    int,
    Field(
        ge=1,
        le=1000,
        description="返す候補の最大数（1〜1000）",
    ),
]


class Suggestion(BaseModel):
    """補完候補を表す値オブジェクト（Value Object）。

    ドメイン駆動設計（DDD）に基づき、補完候補の不変性と
    一貫性を保証する。
    """

    text: str = Field(min_length=1, description="補完テキスト")
    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(description="スコア（0〜1）")

    model_config = {"frozen": True}  # イミュータブル


class SuggestionList(BaseModel):
    """補完候補のコレクション。

    ドメインロジック（ソート、フィルタリング、top_k選択）を
    カプセル化し、ビジネスルールを一箇所に集約する。
    """

    items: list[Suggestion] = Field(default_factory=list, description="補完候補のリスト")

    def model_post_init(self, __context: Any) -> None:
        """初期化後に自動的にスコアでソート（降順）。"""
        # frozenでない場合のみソート
        object.__setattr__(self, "items", sorted(self.items, key=lambda x: x.score, reverse=True))

    def top_k(self, k: int) -> list[Suggestion]:
        """上位k件の候補を取得する。

        Args:
            k: 取得する候補の数

        Returns:
            スコアの高い順にk件の候補
        """
        return self.items[:k]

    def filter_by_score(self, min_score: float) -> "SuggestionList":
        """スコアでフィルタリングする。

        Args:
            min_score: 最小スコア

        Returns:
            フィルタリングされた新しいSuggestionList
        """
        filtered = [s for s in self.items if s.score >= min_score]
        return SuggestionList(items=filtered)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """後方互換性のために辞書リストに変換する。

        Returns:
            {'text': str, 'score': float} 形式の辞書のリスト
        """
        return [s.model_dump() for s in self.items]

    def __len__(self) -> int:
        """候補の数を返す。"""
        return len(self.items)

    def __getitem__(self, index: int) -> Suggestion:
        """インデックスで候補を取得する。"""
        return self.items[index]
