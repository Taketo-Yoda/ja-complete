"""複数の独立した補完メソッドを提供するメイン補完クラス。

このモジュールは、補完モデルの複雑なサブシステムへの
統一されたインターフェースを提供するためにFacadeパターンを実装する。
"""

from pathlib import Path

from pydantic import validate_call

from ja_complete import tokenizer
from ja_complete.models.ngram import NgramModel
from ja_complete.models.phrase import PhraseModel
from ja_complete.models.simple import SimpleDictModel
from ja_complete.types import MorphToken, NgramData, SimpleSuggestions, SuggestionList, TopK


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
        self,
        input_text: str,
        top_k: int = 10,
        fallback_to_ngram: bool | None = None,
        extend_particles: bool = True,
    ) -> SuggestionList:
        """
        オプションのN-gramフォールバック付きでフレーズモデルから補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数
            fallback_to_ngram: デフォルトのフォールバック動作を上書き。
                             Noneの場合、インスタンス設定を使用。
            extend_particles: N-gramフォールバック時に助詞で終わる候補に次の語を追加するか
                            （デフォルト: True）

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
            results = self._ngram_model.suggest(input_text, top_k, extend_particles)

        return results

    # N-gramメソッド
    def load_ngram_model(self, model_path: str | Path) -> None:
        """
        カスタムN-gramモデルを読み込む。

        Args:
            model_path: pickle化されたN-gramモデルファイルへのパス

        Raises:
            FileNotFoundError: モデルファイルが存在しない場合
            ValueError: ファイルがディレクトリの場合
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_path.is_dir():
            raise ValueError(f"Expected file, got directory: {model_path}")

        self._ngram_model = NgramModel(str(model_path))

    @validate_call
    def suggest_from_ngram(
        self, input_text: str, top_k: TopK = 10, extend_particles: bool = True
    ) -> SuggestionList:
        """
        N-gramモデルのみから補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数（1〜1000）
            extend_particles: 助詞で終わる候補に次の語を追加するか（デフォルト: True）

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        Raises:
            ValidationError: top_kが1〜1000の範囲外の場合
        """
        return self._ngram_model.suggest(input_text, top_k, extend_particles)

    # 単純辞書メソッド
    def add_simple_suggestions(self, suggestions: dict[str, list[str]] | SimpleSuggestions) -> None:
        """単純なプレフィックスから補完へのマッピングを追加する。

        Args:
            suggestions: プレフィックス -> 補完リストのマッピング辞書、
                        またはSimpleSuggestions値オブジェクト

        Example:
            >>> completer = JaCompleter()
            >>> # dict形式
            >>> completer.add_simple_suggestions({
            ...     "お": ["おはよう", "おやすみ", "お疲れ様"],
            ...     "あり": ["ありがとう", "ありがとうございます"]
            ... })
            >>> # SimpleSuggestions形式
            >>> phrases = ["今日はいい天気", "今日は雨"]
            >>> simple_sugg = JaCompleter.phrases_to_simple_suggestions(phrases)
            >>> completer.add_simple_suggestions(simple_sugg)
        """
        self._simple_model.add_suggestions(suggestions)

    def suggest_from_simple(
        self,
        input_text: str,
        top_k: int = 10,
        fallback_to_ngram: bool | None = None,
        extend_particles: bool = True,
    ) -> SuggestionList:
        """
        オプションのN-gramフォールバック付きで単純辞書から補完を取得する。

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数
            fallback_to_ngram: デフォルトのフォールバック動作を上書き。
                             Noneの場合、インスタンス設定を使用。
            extend_particles: N-gramフォールバック時に助詞で終わる候補に次の語を追加するか
                            （デフォルト: True）

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
            results = self._ngram_model.suggest(input_text, top_k, extend_particles)

        return results

    # ユーティリティメソッド
    @staticmethod
    def phrases_to_simple_suggestions(
        phrases: list[str],
        min_prefix_length: int = 1,
        max_prefix_length: int = 10,
    ) -> SimpleSuggestions:
        """フレーズから文字ベースのプレフィックスマッピングを生成する。

        各フレーズから1文字〜max_prefix_length文字までのプレフィックスを
        抽出し、SimpleSuggestions形式に変換する。

        Args:
            phrases: 日本語フレーズのリスト
            min_prefix_length: 最小プレフィックス長（デフォルト: 1）
            max_prefix_length: 最大プレフィックス長（デフォルト: 10）

        Returns:
            SimpleSuggestions: プレフィックス -> フレーズリストのマッピング

        Example:
            >>> phrases = ["今日はいい天気", "今日は雨"]
            >>> suggestions = JaCompleter.phrases_to_simple_suggestions(phrases)
            >>> "今" in suggestions.data
            True
            >>> "今日" in suggestions.data
            True
            >>> len(suggestions.data["今"])
            2
        """
        prefix_map: dict[str, set[str]] = {}

        for phrase in phrases:
            for i in range(min_prefix_length, min(max_prefix_length + 1, len(phrase) + 1)):
                prefix = phrase[:i]
                if prefix not in prefix_map:
                    prefix_map[prefix] = set()
                prefix_map[prefix].add(phrase)

        # setをlistに変換してソート
        data = {k: sorted(list(v)) for k, v in prefix_map.items()}
        return SimpleSuggestions(data=data)

    @staticmethod
    def phrases_to_ngram_data(phrases: list[str]) -> NgramData:
        """フレーズをN-gramデータに変換する（形態素情報含む）。

        各フレーズを形態素解析し、unigram/bigram/trigramのカウントと
        形態素情報を抽出する。

        Args:
            phrases: 日本語フレーズのリスト

        Returns:
            NgramData: N-gramカウントと形態素情報を含むデータ

        Example:
            >>> phrases = ["今日はいい天気", "今日は雨"]
            >>> ngram_data = JaCompleter.phrases_to_ngram_data(phrases)
            >>> "今日" in ngram_data.unigrams
            True
            >>> "は" in ngram_data.unigrams
            True
            >>> "今日" in ngram_data.morphology
            True
        """
        from collections import defaultdict

        unigrams: dict[str, int] = {}
        bigrams: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
        trigrams: defaultdict[tuple[str, str], defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        morphology: dict[str, MorphToken] = {}

        for phrase in phrases:
            morphemes = tokenizer.get_morphemes(phrase)
            tokens = [m["surface"] for m in morphemes]

            # unigramカウント
            for token in tokens:
                unigrams[token] = unigrams.get(token, 0) + 1

            # bigramカウント
            for i in range(len(tokens) - 1):
                bigrams[tokens[i]][tokens[i + 1]] += 1

            # trigramカウント
            for i in range(len(tokens) - 2):
                key = (tokens[i], tokens[i + 1])
                trigrams[key][tokens[i + 2]] += 1

            # 形態素情報保存（重複時は最初の出現を保持）
            for morph in morphemes:
                surface = morph["surface"]
                if surface not in morphology:
                    morphology[surface] = MorphToken(
                        surface=surface,
                        pos=morph["pos"],
                        base_form=morph["base_form"],
                    )

        # defaultdictを通常のdictに変換
        return NgramData(
            unigrams=unigrams,
            bigrams={k: dict(v) for k, v in bigrams.items()},
            trigrams={(k1, k2): dict(v) for (k1, k2), v in trigrams.items()},
            morphology=morphology,
        )
