"""形態素解析を用いたフレーズベース補完モデル。

このモデルは自動プレフィックス生成とハイブリッドスコアリングを使用して、
カスタムフレーズリストから補完候補を生成する。
"""

from collections import defaultdict

from pydantic import validate_call

from ja_complete import tokenizer
from ja_complete.models.base import CompletionModel
from ja_complete.types import Suggestion, SuggestionList, TopK

# スコアリング重み（ユースケースに応じて調整可能）
PREFIX_WEIGHT = 0.6  # プレフィックスマッチング品質の重み
MORPHEME_WEIGHT = 0.4  # 形態素重複の重み


class PhraseModel(CompletionModel):
    """
    フレーズベース補完モデル。

    このモデルは以下を使用してフレーズから自動的にプレフィックスを生成する:
    1. 文字レベルのプレフィックス（1-3文字）
    2. 形態素境界
    3. 文節境界（ルールベース）

    スコアリングはハイブリッドアルゴリズムを使用:
    - プレフィックスマッチング品質（60%）
    - 形態素重複（40%）
    """

    def __init__(self) -> None:
        """空のフレーズストレージとプレフィックスインデックスを初期化する。"""
        self.phrases: set[str] = set()
        self.prefix_map: dict[str, set[str]] = defaultdict(set)

    def add_phrases(self, phrases: list[str]) -> None:
        """
        フレーズを追加してプレフィックスインデックスを構築する。

        各フレーズに対して:
        1. Janomeを使用してトークン化
        2. 複数の戦略を使用してプレフィックスを生成
        3. プレフィックス -> 完全なフレーズのマッピング

        Args:
            phrases: 追加する日本語フレーズのリスト

        Example:
            >>> model = PhraseModel()
            >>> model.add_phrases([
            ...     "スマホの買い換えと合わせて一式揃えたい",
            ...     "新生活に備えた準備を始めたい"
            ... ])
        """
        for phrase in phrases:
            if not phrase:
                continue

            self.phrases.add(phrase)
            prefixes = self._generate_prefixes(phrase)

            for prefix in prefixes:
                self.prefix_map[prefix].add(phrase)

    def _generate_prefixes(self, phrase: str) -> set[str]:
        """
        補完のための有用なプレフィックスを全て生成する。

        戦略:
        a. 文字レベル: 最初の1, 2, 3文字
        b. 形態素境界: 各形態素の後
        c. 文節境界: 各文節の後

        Args:
            phrase: プレフィックスを生成するフレーズ

        Returns:
            ユニークなプレフィックスのセット
        """
        prefixes = set()

        # 戦略A: 文字レベルのプレフィックス（1-3文字）
        for i in range(1, min(4, len(phrase) + 1)):
            prefixes.add(phrase[:i])

        # 戦略B: 形態素境界
        tokens = tokenizer.tokenize(phrase)
        position = 0
        for token in tokens:
            position += len(token)
            if position < len(phrase):
                prefixes.add(phrase[:position])

        # 戦略C: 文節境界
        bunsetsu_list = tokenizer.extract_bunsetsu(phrase)
        position = 0
        for bunsetsu in bunsetsu_list[:-1]:  # 最後の文節を除く（完全なフレーズになる）
            position += len(bunsetsu)
            if position < len(phrase):
                prefixes.add(phrase[:position])

        return prefixes

    def _calculate_score(self, input_text: str, phrase: str) -> float:
        """
        プレフィックスマッチングと形態素重複を組み合わせたハイブリッドスコアを計算する。

        コンポーネント1: プレフィックスマッチングスコア（0.0 ~ 0.6）
        - 完全一致: 0.6
        - 部分一致: 0.3 + (比率 * 0.3)

        コンポーネント2: 形態素重複スコア（0.0 ~ 0.4）
        - 形態素のJaccard類似度に基づく

        Args:
            input_text: ユーザー入力
            phrase: 候補フレーズ

        Returns:
            [0, 1]の範囲の合計スコア
        """
        # コンポーネント1: プレフィックスマッチングスコア
        if phrase == input_text:
            prefix_score = PREFIX_WEIGHT  # 完全一致
        elif phrase.startswith(input_text):
            ratio = len(input_text) / len(phrase)
            prefix_score = (PREFIX_WEIGHT / 2) + (ratio * (PREFIX_WEIGHT / 2))
        else:
            return 0.0  # プレフィックスマッチなし

        # コンポーネント2: 形態素重複スコア
        input_morphemes = set(tokenizer.tokenize(input_text))
        phrase_morphemes = set(tokenizer.tokenize(phrase))

        if not input_morphemes:
            morpheme_score = 0.0
        else:
            # Jaccard類似度を計算
            intersection = input_morphemes & phrase_morphemes
            morpheme_overlap_ratio = len(intersection) / len(input_morphemes)
            morpheme_score = morpheme_overlap_ratio * MORPHEME_WEIGHT

        # 合計スコア
        total_score = prefix_score + morpheme_score
        return min(total_score, 1.0)  # 1.0で上限

    @validate_call
    def suggest(self, input_text: str, top_k: TopK = 10) -> SuggestionList:
        """
        input_textにプレフィックスとしてマッチするフレーズを検索する。

        以下を組み合わせたハイブリッドスコアリングアルゴリズムを使用:
        1. プレフィックスマッチング品質（60%重み）
        2. 形態素重複（40%重み）

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

        # 候補フレーズを検索
        candidates = set()

        # まず完全なプレフィックスで検索
        if input_text in self.prefix_map:
            candidates.update(self.prefix_map[input_text])

        # 全フレーズについてもプレフィックスマッチをチェック
        # （プレフィックスが索引されていない場合のため）
        for phrase in self.phrases:
            if phrase.startswith(input_text):
                candidates.add(phrase)

        # 候補をスコアリングしてランク付け
        suggestions: list[Suggestion] = []
        for phrase in candidates:
            score = self._calculate_score(input_text, phrase)
            if score > 0:
                suggestions.append(Suggestion(text=phrase, score=score))

        # SuggestionListでラップ（自動的にスコアでソートされる）してtop_kを返す
        suggestion_list = SuggestionList(items=suggestions)
        return SuggestionList(items=suggestion_list.top_k(top_k))
