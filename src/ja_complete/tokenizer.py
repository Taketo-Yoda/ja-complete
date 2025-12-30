"""Janomeを使用した日本語テキストのトークン化。

このモジュールはJanomeトークナイザーのラッパーを提供し、
形態素解析と文節抽出のための便利なメソッドを提供する。
"""

from typing import Any

from janome.tokenizer import Tokenizer

# モジュールレベルのシングルトン（Singletonパターン）
_tokenizer_instance: Tokenizer | None = None


def _get_tokenizer() -> Tokenizer:
    """トークナイザーインスタンスを取得または作成（Singletonパターン）。"""
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = Tokenizer()
    return _tokenizer_instance


def tokenize(text: str) -> list[str]:
    """
    日本語テキストを形態素に分割する。

    Args:
        text: 分割する日本語テキスト

    Returns:
        表層形（形態素）のリスト

    Example:
        >>> tokenize("今日はいい天気")
        ['今日', 'は', 'いい', '天気']
    """
    if not text:
        return []

    tokenizer = _get_tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]  # type: ignore[attr-defined]


def get_morphemes(text: str) -> list[dict[str, str]]:
    """
    詳細な形態素情報を取得する。

    Args:
        text: 解析する日本語テキスト

    Returns:
        surface, pos（品詞）, base_formのキーを持つ辞書のリスト

    Example:
        >>> get_morphemes("今日は")
        [
            {'surface': '今日', 'pos': '名詞', 'base_form': '今日'},
            {'surface': 'は', 'pos': '助詞', 'base_form': 'は'}
        ]
    """
    if not text:
        return []

    tokenizer = _get_tokenizer()
    morphemes: list[dict[str, str]] = []

    for token in tokenizer.tokenize(text):  # type: ignore[attr-defined]
        # Janomeのpart_of_speechフォーマット: "品詞,品詞細分類1,品詞細分類2,品詞細分類3,..."
        token_any: Any = token  # Pylanceの型エラー回避のための明示的なキャスト
        pos_info = token_any.part_of_speech.split(",")

        morphemes.append(
            {
                "surface": token_any.surface,
                "pos": pos_info[0],  # 主要品詞カテゴリ
                "base_form": token_any.base_form,
            }
        )

    return morphemes


def extract_bunsetsu(text: str) -> list[str]:
    """
    ルールベースのアプローチを使用して文節（フレーズチャンク）を抽出する。

    ルール: 自立語（content word）+ 付属語（function words）

    実装は以下のルールに従う:
    1. 助詞の後 - ただし、次も助詞の場合は除く
    2. 助動詞の後
    3. 自立語（名詞、動詞、形容詞、副詞、連体詞、接続詞）の前
       - ただし、現在の形態素が接頭詞・接尾辞の場合は除く

    Args:
        text: 日本語テキスト

    Returns:
        文節（フレーズチャンク）のリスト

    Example:
        >>> extract_bunsetsu("今日はいい天気ですね")
        ['今日は', 'いい', '天気ですね']
    """
    if not text:
        return []

    morphemes = get_morphemes(text)
    bunsetsu_list: list[str] = []
    current_bunsetsu = ""

    for i, morph in enumerate(morphemes):
        surface = morph["surface"]
        pos = morph["pos"]  # 主要品詞カテゴリ

        current_bunsetsu += surface

        # 現在の文節を終了すべきかチェック
        should_break = False

        # ルール1: 助詞の後 - ただし、次も助詞の場合は除く
        if pos == "助詞":
            next_pos = morphemes[i + 1]["pos"] if i + 1 < len(morphemes) else None
            if next_pos != "助詞":
                should_break = True

        # ルール2: 助動詞の後
        elif pos == "助動詞":
            should_break = True

        # ルール3: 自立語の前
        elif i + 1 < len(morphemes):
            next_pos = morphemes[i + 1]["pos"]
            if next_pos in ["名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞"]:
                # ただし、現在が接頭詞・接尾辞の場合は除く
                if pos not in ["接頭詞", "接尾辞"]:
                    should_break = True

        if should_break or i == len(morphemes) - 1:
            if current_bunsetsu:
                bunsetsu_list.append(current_bunsetsu)
                current_bunsetsu = ""

    return bunsetsu_list
