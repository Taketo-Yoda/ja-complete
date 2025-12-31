"""カスタム型定義。

このモジュールは、ドメイン固有の型とバリデーションを提供する。
"""

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

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


class SimpleSuggestions(BaseModel):
    """単純辞書補完用の値クラス。

    プレフィックスから補完候補へのマッピングを保持する。
    """

    data: dict[str, list[str]] = Field(
        default_factory=dict, description="プレフィックス -> 補完候補リストのマッピング"
    )

    model_config = {"frozen": True}  # イミュータブル

    @field_validator("data")
    @classmethod
    def validate_suggestions(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        """単純辞書補完データのバリデーション。

        Args:
            v: プレフィックス -> 補完候補リストのマッピング

        Returns:
            検証済みのマッピング

        Raises:
            ValueError: 空文字列のキー、空リスト、またはリスト内の空文字列が含まれる場合
        """
        for key, values in v.items():
            if not key:
                raise ValueError("Empty string keys are not allowed")
            if not values:
                raise ValueError(f"Empty list for key '{key}' is not allowed")
            for value in values:
                if not value:
                    raise ValueError(f"Empty string in values list for key '{key}' is not allowed")
        return v

    def to_dict(self) -> dict[str, list[str]]:
        """後方互換性のために通常の辞書に変換する。

        Returns:
            プレフィックス -> 補完候補リストのマッピング辞書
        """
        return self.data.copy()


class MorphToken(BaseModel):
    """形態素トークン（表層形、品詞、基本形）。

    Janomeの形態素解析結果を保持する値オブジェクト。
    """

    surface: str = Field(min_length=1, description="表層形")
    pos: str = Field(min_length=1, description="品詞")
    base_form: str = Field(min_length=1, description="基本形")

    model_config = {"frozen": True}  # イミュータブル


class NgramData(BaseModel):
    """N-gramカウントデータ（形態素情報付き）。

    フレーズから抽出されたN-gram統計と形態素情報を保持する。
    """

    unigrams: dict[str, int] = Field(
        default_factory=dict, description="unigram -> カウントのマッピング"
    )
    bigrams: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="token1 -> {token2 -> カウント}のマッピング"
    )
    trigrams: dict[tuple[str, str], dict[str, int]] = Field(
        default_factory=dict, description="(token1, token2) -> {token3 -> カウント}のマッピング"
    )
    morphology: dict[str, MorphToken] = Field(
        default_factory=dict, description="トークン -> 形態素情報のマッピング"
    )

    @field_validator("unigrams", "bigrams", "trigrams")
    @classmethod
    def validate_counts(cls, v: Any) -> Any:
        """N-gramカウントのバリデーション。

        Args:
            v: カウントデータ

        Returns:
            検証済みのカウントデータ

        Raises:
            ValueError: 負またはゼロのカウントが含まれる場合
        """
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, int):
                    if value <= 0:
                        raise ValueError(f"Count must be positive, got {value} for key {key}")
                elif isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if nested_value <= 0:
                            raise ValueError(
                                f"Count must be positive, got {nested_value} "
                                f"for key {key} -> {nested_key}"
                            )
        return v
