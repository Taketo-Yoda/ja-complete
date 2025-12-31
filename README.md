# ja-complete

LLMやデータベースを使わない、軽量なオフライン日本語入力補完ライブラリ

## 概要

`ja-complete`は日本語テキスト補完・予測のための純粋なPython OSSライブラリです。異なるユースケースに対応する複数の独立した補完メソッドを提供します：

- **フレーズベース補完**: 自動プレフィックス生成を伴うカスタムフレーズリスト
- **N-gramモデル**: 日本語テキストコーパスに基づく統計的予測
- **単純辞書**: プレフィックスから候補への直接マッピング
- **カスタムJSONL**: 柔軟なカスタムデータフォーマットサポート

主な特徴:
- LLMやデータベース不要: 完全オフラインで軽量
- 複数の独立した補完API
- Janomeによる形態素解析
- CLIツール、エディタ、Webアプリケーションへの簡単な統合

## インストール

```bash
pip install ja-complete
# または uv を使用
uv add ja-complete
```

## クイックスタート

### フレーズベース補完

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# カスタムフレーズを追加
phrases = [
    "スマホの買い換えと合わせて一式揃えたい",
    "新生活に備えた準備を始めたい",
    "夏を爽やかに過ごしたい",
]
completer.add_phrases(phrases)

# 補完を取得（フレーズマッチがない場合、自動的にN-gramにフォールバック）
results = completer.suggest_from_phrases("ス", top_k=5)
print(results)  # [{'text': 'スマホの買い換えと合わせて一式揃えたい', 'score': 0.82}, ...]

# 厳密なフレーズマッチングのためN-gramフォールバックを無効化
results = completer.suggest_from_phrases("未登録の入力", fallback_to_ngram=False)
print(results)  # [] (フレーズマッチがない場合は空)
```

### N-gram補完

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# デフォルトのN-gramモデルを使用
results = completer.suggest_from_ngram("今日は", top_k=5)
print(results)  # [{'text': '今日はいい天気', 'score': 0.85}, ...]

# 助詞拡張を無効化
results = completer.suggest_from_ngram("今日は", top_k=5, extend_particles=False)
print(results)  # 助詞で終わる候補のみ
```

### 単純辞書補完

```python
from ja_complete import JaCompleter

completer = JaCompleter()

# 単純なプレフィックスマッピングを追加
suggestions = {
    "お": ["おはよう", "おやすみ", "お疲れ様"],
    "あり": ["ありがとう", "ありがとうございます"],
}
completer.add_simple_suggestions(suggestions)

# 補完を取得（マッチがない場合、自動的にN-gramにフォールバック）
results = completer.suggest_from_simple("あり", top_k=3)
print(results)  # [{'text': 'ありがとう', 'score': 1.0}, ...]

# フォールバックを無効化
results = completer.suggest_from_simple("未登録", fallback_to_ngram=False)
print(results)  # [] (マッチがない場合は空)
```

### フレーズをJSONLに変換

```python
from ja_complete import JaCompleter

# N-gramモデルトレーニング用にフレーズをJSONL形式に変換
phrases = [
    "今日はいい天気ですね",
    "明日は雨が降りそうです",
    "週末は晴れるといいな",
]

jsonl = JaCompleter.convert_to_jsonl(phrases)
print(jsonl)
# 出力:
# {"text": "今日はいい天気ですね", "tokens": ["今日", "は", "いい", "天気", "です", "ね"]}
# {"text": "明日は雨が降りそうです", "tokens": ["明日", "は", "雨", "が", "降り", "そう", "です"]}
# {"text": "週末は晴れるといいな", "tokens": ["週末", "は", "晴れる", "と", "いい", "な"]}

# モデルトレーニング用にファイルに保存
with open("phrases.jsonl", "w", encoding="utf-8") as f:
    f.write(jsonl)
```

## CLI使用方法

```bash
# フレーズベース補完
ja-complete phrase "新生活" --phrases phrases.txt

# N-gram補完
ja-complete ngram "今日は"

# 単純辞書補完
ja-complete simple "あり" --dict suggestions.json
```

## APIリファレンス

### JaCompleter

複数の補完メソッドを提供するメインクラス。

#### コンストラクタ

- `JaCompleter(enable_ngram_fallback: bool = True)`
  - オプションのN-gramフォールバック付きで補完器を初期化
  - `enable_ngram_fallback=True`の場合、フレーズおよび単純辞書メソッドはマッチが見つからない場合に自動的にN-gram補完を使用

#### メソッド

**フレーズベース補完:**

- `add_phrases(phrases: List[str]) -> None`
  - フレーズベース補完用のフレーズを追加
  - 形態素解析を使用して自動的にプレフィックスを生成

- `suggest_from_phrases(input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None, extend_particles: bool = True) -> List[Dict[str, Any]]`
  - 追加されたフレーズから補完を取得
  - マッチがなく`fallback_to_ngram=True`（またはインスタンスデフォルト）の場合、N-gram補完を返す
  - スコア付きのランク付けされた結果を返す

**N-gram補完:**

- `suggest_from_ngram(input_text: str, top_k: int = 10, extend_particles: bool = True) -> List[Dict[str, Any]]`
  - N-gramモデルを使用して補完を取得
  - デフォルトモデルまたはロードされたカスタムモデルを使用
  - `extend_particles=True`の場合、助詞で終わる候補に次の語を自動追加

- `load_ngram_model(model_path: str) -> None`
  - ファイルからカスタムN-gramモデルを読み込む

**単純辞書補完:**

- `add_simple_suggestions(suggestions: Dict[str, List[str]]) -> None`
  - プレフィックスから候補へのマッピングを追加

- `suggest_from_simple(input_text: str, top_k: int = 10, fallback_to_ngram: bool | None = None, extend_particles: bool = True) -> List[Dict[str, Any]]`
  - 単純辞書から補完を取得
  - マッチがなく`fallback_to_ngram=True`（またはインスタンスデフォルト）の場合、N-gram補完を返す
  - 直接プレフィックスマッチング

**ユーティリティメソッド:**

- `convert_to_jsonl(phrases: List[str]) -> str` (静的メソッド)
  - フレーズのリストをJSONL形式に変換
  - 各行には`{"text": "phrase", "tokens": ["token1", "token2", ...]}`が含まれる
  - N-gramモデル用のトレーニングデータ準備に便利

## スコアリングの仕組み

### フレーズベース補完のスコアリング

フレーズベース補完は、プレフィックスマッチングと意味的類似性の両方を考慮したハイブリッドスコアリングアルゴリズムを使用します：

**スコア構成要素:**
1. **プレフィックスマッチ品質（60%）**: 入力がフレーズの先頭とどれだけよくマッチするか
2. **形態素オーバーラップ（40%）**: 入力からの形態素（単語単位）がフレーズにいくつ現れるか

**例:**

```python
# 完全な形態素オーバーラップを持つ長い入力
入力: "スマホの買い換え"
フレーズ: "スマホの買い換えと合わせて一式揃えたい"
スコア: 0.82 (高 - 良好なプレフィックスマッチ + すべての形態素が存在)

# 短い入力
入力: "スマホ"
フレーズ: "スマホの買い換えと合わせて一式揃えたい"
スコア: 0.75 (良好 - 短いプレフィックスだが形態素は存在)

# 完全一致
入力: "夏を爽やかに過ごしたい"
フレーズ: "夏を爽やかに過ごしたい"
スコア: 1.0 (完全一致)
```

このハイブリッドアプローチにより：
- 入力と正確に始まる補完が優先される
- 意味的に関連するフレーズ（キーワードを含む）がより高くランク付けされる
- より短く関連性の高い補完が不当にペナルティを受けない

### N-gram補完の助詞拡張機能

N-gram補完は助詞で終わる候補を自動的に拡張する機能を持っています：

**動作:**
- 「今日は」のような助詞で終わる入力に対し、次に来る可能性の高い1〜3語を予測
- 元のスコアと次のトークンの確率を掛け合わせて総合スコアを算出
- 元の助詞終わりの候補と拡張された候補の両方を返す

**例:**

```python
入力: "今日は"
結果（extend_particles=True）:
  - "今日は " (元の候補)
  - "今日は晴れ" (拡張された候補)
  - "今日は雨" (拡張された候補)
  - "今日は良い天気" (拡張された候補)

結果（extend_particles=False）:
  - "今日は " (元の候補のみ)
```

## セキュリティに関する考慮事項

**重要: Pickleセキュリティ警告**

N-gramモデルはPythonの`pickle`モジュールを使用してシリアライズされています。**Pickleファイルは読み込み時に任意のコードを実行できます。**

- ⚠️ 信頼できるソースからのモデルファイルのみを読み込んでください
- ⚠️ 不明または信頼できない出所の`.pkl`ファイルを読み込まないでください
- ⚠️ カスタムモデルを読み込むとセキュリティ警告が表示されます

```python
# 安全: デフォルトモデルの使用（パッケージに含まれる）
completer = JaCompleter()

# 警告: カスタムモデルの読み込み（信頼できるファイルのみ使用！）
completer.load_ngram_model("custom_model.pkl")  # セキュリティ警告が表示されます
```

詳細については、[DEVELOPING.md](DEVELOPING.md#セキュリティ)を参照してください。

## カスタムN-gramモデルの構築

独自のN-gramモデルを構築したい上級ユーザー向け：

```bash
# Wikipedia抽出用にgensimをインストール
pip install gensim

# 日本語Wikipediaダンプをダウンロード
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2

# gensimを使用してテキストを抽出（Python 3.13+推奨）
python extract_wiki_text.py

# またはWikiExtractorを使用（Python 3.12以前）
# python -m wikiextractor.WikiExtractor jawiki-latest-pages-articles.xml.bz2 -o wiki_text/

# N-gramモデルを構築
python scripts/build_ngram_model.py --input training_data/ --output my_model.pkl --verbose
```

## Git LFS（Large File Storage）

このプロジェクトでは、モデルファイル（`*.pkl`）をGit LFSで管理しています。

### 開発者向け

リポジトリをクローンまたはプルする際に、Git LFSがインストールされていることを確認してください：

```bash
# Git LFSのインストール確認
git lfs version

# インストールされていない場合
# macOS:
brew install git-lfs

# Ubuntu/Debian:
sudo apt-get install git-lfs

# リポジトリでGit LFSを有効化
git lfs install
```

詳細なセットアップ手順については、[docs/GIT_LFS_SETUP.md](docs/GIT_LFS_SETUP.md)を参照してください。

### ユーザー向け

通常の`pip install ja-complete`または`uv add ja-complete`では、PyPIから自動的にパッケージがインストールされます。Git LFSは不要です。

## コントリビューション

コントリビューションを歓迎します！開発セットアップとガイドラインについては、[DEVELOPING.md](DEVELOPING.md)を参照してください。

## ライセンス

このプロジェクトは**デュアルライセンス**方式を採用しています：

### ソースコード - MITライセンス

ソースコードは[MITライセンス](LICENSE)の下でライセンスされています。

```
Copyright (c) 2025 Taketo Yoda
```

### N-gramモデルデータ - CC BY-SA 3.0

N-gramモデルファイル（`*.pkl`）は[CC BY-SA 3.0](LICENSE-CC-BY-SA.txt)の下でライセンスされています。

デフォルトモデルは[日本語Wikipedia](https://ja.wikipedia.org/)データでトレーニングされています：
- ソース: 日本語Wikipedia
- ライセンス: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- 著作権: © Wikipediaコントリビューター
- データセット: 約10万記事のサブセット

**重要**: N-gramモデルを配布または修正する場合、以下を行う必要があります：
1. Wikipediaへの帰属表示を提供
2. 行われた変更を示す
3. CC BY-SA 3.0または互換性のあるライセンスの下で配布

詳細については、[LICENSE-CC-BY-SA.txt](LICENSE-CC-BY-SA.txt)を参照してください。
