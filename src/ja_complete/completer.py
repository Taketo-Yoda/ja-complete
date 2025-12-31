"""複数の独立した補完メソッドを提供するメイン補完クラス。

このモジュールは、補完モデルの複雑なサブシステムへの
統一されたインターフェースを提供するためにFacadeパターンを実装する。
"""

import json

from pydantic import validate_call

from ja_complete import tokenizer
from ja_complete.models.ngram import NgramModel
from ja_complete.models.phrase import PhraseModel
from ja_complete.models.simple import SimpleDictModel
from ja_complete.types import SuggestionList, TopK


class JaCompleter:
    """
    複数の独立した補完メソッドを提供するメイン補完クラス。

    このクラスはFacadeとして機能し、トークナイザーと複数の補完モデルの
    複雑なサブシステムへの簡潔なインターフェースを提供する。

    サポート機能:
    - 自動プレフィックス生成によるフレーズベース補完
    - N-gram統計補完
    - 単純辞書補完
    - フレーズおよび単純辞書メソッド用のN-gramフォールバック
    """

    def __init__(self, enable_ngram_fallback: bool = True) -> None:
        """
        JaCompleterを初期化する。

        Args:
            enable_ngram_fallback: Trueの場合、フレーズベースおよび単純辞書補完は
                                  マッチが見つからないときにN-gramにフォールバックする
                                  （デフォルト: True）
        """
        self._phrase_model = PhraseModel()
        self._ngram_model = NgramModel()  # デフォルトモデルを読み込む
        self._simple_model = SimpleDictModel()
        self._enable_ngram_fallback = enable_ngram_fallback

    # フレーズベースメソッド
    def add_phrases(self, phrases: list[str]) -> None:
        """
        フレーズベース補完にフレーズを追加する。

        Args:
            phrases: 日本語フレーズのリスト

        Example:
            >>> completer = JaCompleter()
            >>> completer.add_phrases([
            ...     "スマホの買い換えと合わせて一式揃えたい",
            ...     "新生活に備えた準備を始めたい"
            ... ])
        """
        self._phrase_model.add_phrases(phrases)

    def suggest_from_phrases(
        self, input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None
    ) -> SuggestionList:
        """
        オプションのN-gramフォールバック付きでフレーズモデルから補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数
            fallback_to_ngram: デフォルトのフォールバック動作を上書き。
                             Noneの場合、インスタンス設定を使用。

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        動作:
            1. フレーズモデルから補完の取得を試みる
            2. マッチがなくフォールバックが有効な場合、N-gramモデルを使用
            3. スコアでソートされたtop_k個の結果を返す
        """
        results = self._phrase_model.suggest(input_text, top_k)

        # 有効で結果がない場合はN-gramにフォールバック
        use_fallback = (
            fallback_to_ngram if fallback_to_ngram is not None else self._enable_ngram_fallback
        )

        if not results and use_fallback:
            results = self._ngram_model.suggest(input_text, top_k)

        return results

    # N-gramメソッド
    def load_ngram_model(self, model_path: str) -> None:
        """
        カスタムN-gramモデルを読み込む。

        Args:
            model_path: pickle化されたN-gramモデルファイルへのパス
        """
        self._ngram_model = NgramModel(model_path)

    @validate_call
    def suggest_from_ngram(self, input_text: str, top_k: TopK = 10) -> SuggestionList:
        """
        N-gramモデルのみから補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数（1〜1000）

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        Raises:
            ValidationError: top_kが1〜1000の範囲外の場合
        """
        return self._ngram_model.suggest(input_text, top_k)

    # 単純辞書メソッド
    def add_simple_suggestions(self, suggestions: dict[str, list[str]]) -> None:
        """
        単純なプレフィックスから補完へのマッピングを追加する。

        Args:
            suggestions: プレフィックス -> 補完リストのマッピング辞書

        Example:
            >>> completer = JaCompleter()
            >>> completer.add_simple_suggestions({
            ...     "お": ["おはよう", "おやすみ", "お疲れ様"],
            ...     "あり": ["ありがとう", "ありがとうございます"]
            ... })
        """
        self._simple_model.add_suggestions(suggestions)

    def suggest_from_simple(
        self, input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None
    ) -> SuggestionList:
        """
        オプションのN-gramフォールバック付きで単純辞書から補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数
            fallback_to_ngram: デフォルトのフォールバック動作を上書き。
                             Noneの場合、インスタンス設定を使用。

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        動作:
            1. 単純辞書から補完の取得を試みる
            2. マッチがなくフォールバックが有効な場合、N-gramモデルを使用
            3. スコアでソートされたtop_k個の結果を返す
        """
        results = self._simple_model.suggest(input_text, top_k)

        # 有効で結果がない場合はN-gramにフォールバック
        use_fallback = (
            fallback_to_ngram if fallback_to_ngram is not None else self._enable_ngram_fallback
        )

        if not results and use_fallback:
            results = self._ngram_model.suggest(input_text, top_k)

        return results

    # ユーティリティメソッド
    @staticmethod
    def convert_to_jsonl(phrases: list[str]) -> str:
        """
        フレーズのリストをN-gramモデルトレーニング用のJSONL形式に変換する。

        各フレーズはメタデータを持つJSONオブジェクトとなり、
        カスタムN-gramモデルの構築やトレーニングデータに使用できる。

        Args:
            phrases: 日本語フレーズのリスト

        Returns:
            JSONL文字列（1行に1つのJSONオブジェクト）

        Example:
            >>> phrases = ["今日はいい天気", "明日は雨"]
            >>> jsonl = JaCompleter.convert_to_jsonl(phrases)
            >>> print(jsonl)
            {"text": "今日はいい天気", "tokens": ["今日", "は", "いい", "天気"]}
            {"text": "明日は雨", "tokens": ["明日", "は", "雨"]}

        Note:
            JSONL出力には、より簡単なN-gramモデルトレーニングのために
            フレーズのトークン化バージョンが含まれる。
        """
        lines: list[str] = []
        for phrase in phrases:
            tokens = tokenizer.tokenize(phrase)
            obj = {"text": phrase, "tokens": tokens}
            lines.append(json.dumps(obj, ensure_ascii=False))

        return "\n".join(lines)
