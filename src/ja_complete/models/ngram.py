"""N-gram統計補完モデル。

このモデルはbigramとtrigram統計を使用して、
ユーザー入力の可能性の高い継続を予測する。
"""

import os
import pickle
import warnings
from pathlib import Path

from pydantic import validate_call

from ja_complete import tokenizer
from ja_complete.models.base import CompletionModel
from ja_complete.types import MorphToken, NgramData, Suggestion, SuggestionList, TopK

# Laplaceスムージングパラメータ
SMOOTHING_ALPHA = 1.0


class NgramModel(CompletionModel):
    """
    N-gram統計補完モデル。

    bigramとtrigram確率を使用して補完を生成する。
    より良い確率推定のためLaplaceスムージングを実装。
    """

    def __init__(self, model_path: str | None = None, skip_default: bool = False) -> None:
        """
        N-gramモデルを初期化する。

        Args:
            model_path: pickleモデルファイルへのパス。
                       Noneの場合、パッケージデータからデフォルトモデルを読み込む。
            skip_default: Trueの場合、デフォルトモデルのロードをスキップ（テスト用）。
                         環境変数SKIP_DEFAULT_MODELが設定されている場合も同様。
        """
        self.unigrams: dict[str, int] = {}
        self.bigrams: dict[str, dict[str, int]] = {}
        self.trigrams: dict[tuple[str, str], dict[str, int]] = {}
        self.morphology: dict[str, MorphToken] = {}
        self.vocabulary_size: int = 0

        # テストモード用の環境変数をチェック
        skip_default = skip_default or os.getenv("SKIP_DEFAULT_MODEL") == "1"

        if model_path:
            self.load_model(model_path)
        elif not skip_default:
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

        # デフォルトモデル以外を読み込む場合、pickleのセキュリティリスクを警告
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

        # モデル構造を検証
        if not isinstance(model, dict):
            raise ValueError(f"Invalid model format: expected dict, got {type(model)}")

        # モデルに期待されるキーが欠けている場合は警告（ただし読み込みは許可）
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
        self.morphology = model.get("morphology", {})  # 後方互換性: オプショナル
        self.vocabulary_size = len(self.unigrams)

    def load_default_model(self) -> None:
        """パッケージデータからデフォルトのN-gramモデルを読み込む。"""
        default_model_path = Path(__file__).parent.parent / "data" / "default_ngram.pkl"
        if default_model_path.exists():
            self.load_model(str(default_model_path))
        else:
            # デフォルトモデルが利用できない場合 - 空のモデルを使用
            self.vocabulary_size = 0

    def add_ngram_data(self, data: NgramData) -> None:
        """N-gramデータをモデルにマージする（破壊的更新）。

        既存のカウントに新しいカウントを加算し、形態素情報を追加する。

        Args:
            data: マージするN-gramデータ

        Example:
            >>> model = NgramModel(skip_default=True)
            >>> from ja_complete import JaCompleter
            >>> phrases = ["今日はいい天気"]
            >>> ngram_data = JaCompleter.phrases_to_ngram_data(phrases)
            >>> model.add_ngram_data(ngram_data)
            >>> "今日" in model.unigrams
            True
        """
        # unigramマージ
        for token, count in data.unigrams.items():
            self.unigrams[token] = self.unigrams.get(token, 0) + count

        # bigramマージ
        for token1, next_tokens in data.bigrams.items():
            if token1 not in self.bigrams:
                self.bigrams[token1] = {}
            for token2, count in next_tokens.items():
                self.bigrams[token1][token2] = self.bigrams[token1].get(token2, 0) + count

        # trigramマージ
        for (token1, token2), next_tokens in data.trigrams.items():
            key = (token1, token2)
            if key not in self.trigrams:
                self.trigrams[key] = {}
            for token3, count in next_tokens.items():
                self.trigrams[key][token3] = self.trigrams[key].get(token3, 0) + count

        # 形態素情報マージ（既存のトークンには新しい情報で上書きしない）
        for surface, morph_token in data.morphology.items():
            if surface not in self.morphology:
                self.morphology[surface] = morph_token

        # vocabulary_size更新
        self.vocabulary_size = len(self.unigrams)

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

    def _get_next_token_candidates(self, history: list[str]) -> dict[str, float]:
        """
        historyに基づいて次のトークン候補とその確率を取得する。

        Args:
            history: コンテキストトークンのリスト（最大2トークン）

        Returns:
            次のトークン -> 確率のマッピング
        """
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

        return candidates

    def _extend_particle_suggestions(
        self, suggestions: list[Suggestion], max_extensions: int = 3
    ) -> list[Suggestion]:
        """
        助詞で終わる補完候補に次の語を追加する。

        Args:
            suggestions: 拡張する補完候補のリスト
            max_extensions: 各候補に追加する次の語の最大数

        Returns:
            拡張された補完候補のリスト（元の候補も含む）
        """
        extended: list[Suggestion] = []

        for suggestion in suggestions:
            # 元の候補を追加
            extended.append(suggestion)

            # 助詞で終わるかチェック
            if tokenizer.ends_with_particle(suggestion.text):
                # 次のトークンを予測
                tokens = tokenizer.tokenize(suggestion.text)
                if not tokens:
                    continue

                history = tokens[-2:] if len(tokens) >= 2 else tokens[-1:]
                next_candidates = self._get_next_token_candidates(history)

                # 上位の次のトークンを追加
                sorted_candidates = sorted(
                    next_candidates.items(), key=lambda x: x[1], reverse=True
                )
                for next_token, prob in sorted_candidates[:max_extensions]:
                    extended_text = suggestion.text + next_token
                    # 元のスコアと次のトークンのスコアを掛け合わせる
                    combined_score = suggestion.score * prob
                    extended.append(Suggestion(text=extended_text, score=combined_score))

        return extended

    @validate_call
    def suggest(
        self, input_text: str, top_k: TopK = 10, extend_particles: bool = True
    ) -> SuggestionList:
        """
        N-gram確率を使用して次の単語を予測する。

        アルゴリズム:
        1. input_textをトークン化
        2. 最後の1-2トークンをコンテキストとして取得
        3. 可能性のある全ての次のトークンの確率を計算
        4. 可能性の高い次のトークンを追加して補完を生成
        5. extend_particles=Trueの場合、助詞で終わる候補に次の語を追加
        6. 確率でソートされたtop_k個の結果を返す

        Args:
            input_text: ユーザー入力テキスト
            top_k: 候補の最大数（1〜1000）
            extend_particles: 助詞で終わる候補に次の語を追加するか（デフォルト: True）

        Returns:
            SuggestionList: スコアの降順でソート済みの補完候補リスト

        Raises:
            ValidationError: top_kが1〜1000の範囲外の場合
            ValueError: input_textが空の場合
        """
        if not input_text:
            raise ValueError("input_text cannot be empty")

        # 入力をトークン化
        tokens = tokenizer.tokenize(input_text)
        if not tokens:
            return SuggestionList(items=[])

        # コンテキストを取得（最後の1-2トークン）
        history = tokens[-2:] if len(tokens) >= 2 else tokens[-1:]

        # 候補となる次のトークンを取得
        candidates = self._get_next_token_candidates(history)

        # まだ候補がない場合は全unigramを使用
        if not candidates and self.unigrams:
            for next_token in list(self.unigrams.keys())[:50]:  # 多すぎるのを避けるため制限
                prob = self._calculate_probability(history, next_token)
                candidates[next_token] = prob

        # 補完を構築
        suggestions: list[Suggestion] = []
        for next_token, prob in candidates.items():
            completion_text = input_text + next_token
            suggestions.append(Suggestion(text=completion_text, score=prob))

        # 助詞で終わる候補を拡張
        if extend_particles:
            suggestions = self._extend_particle_suggestions(suggestions)

        # 末尾のスペース（全角・半角）を除去し、重複を排除
        dedup_map: dict[str, float] = {}
        for suggestion in suggestions:
            # 末尾のスペース（全角U+3000・半角U+0020）を除去
            stripped_text = suggestion.text.rstrip(" 　")

            # 入力値と一致する場合はスキップ（補完として意味がない）
            if stripped_text == input_text:
                continue

            # 重複する場合は高い方のスコアを採用
            if stripped_text in dedup_map:
                dedup_map[stripped_text] = max(dedup_map[stripped_text], suggestion.score)
            else:
                dedup_map[stripped_text] = suggestion.score

        # Suggestionリストに変換
        suggestions = [Suggestion(text=text, score=score) for text, score in dedup_map.items()]

        # SuggestionListでラップ（自動的にソートされる）してtop_kを返す
        suggestion_list = SuggestionList(items=suggestions)
        return SuggestionList(items=suggestion_list.top_k(top_k))
