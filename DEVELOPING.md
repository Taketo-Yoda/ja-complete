# ja-complete 開発ガイド

このドキュメントは、ja-completeライブラリの開発者向けの情報をまとめています。

## 開発環境のセットアップ

### 必要な環境

- Python 3.10以上
- uv (推奨) または pip

### 初期セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/your-org/ja-complete.git
cd ja-complete

# uvを使用する場合（推奨）
uv sync

# 開発用依存関係を含めてインストール
uv sync --group dev
```

## テスト

### テストの実行

```bash
# 全テストを実行
uv run pytest

# 詳細な出力で実行
uv run pytest -v

# 特定のテストファイルを実行
uv run pytest tests/test_completer.py

# カバレッジレポート付きで実行
uv run pytest --cov=ja_complete --cov-report=html
```

### テストの構造

```
tests/
├── __init__.py
├── models/
│   ├── test_ngram.py      # N-gramモデルのテスト (52 tests)
│   ├── test_phrase.py     # フレーズモデルのテスト (63 tests)
│   └── test_simple.py     # 単純辞書モデルのテスト (57 tests)
├── test_cli.py            # CLIのテスト (36 tests)
├── test_completer.py      # メイン補完クラスのテスト (8 tests)
└── test_tokenizer.py      # トークナイザーのテスト (38 tests)

合計: 209 tests
```

## コード品質チェック

### 重要なルール

**コード生成・修正後は必ず以下のコマンドを実行してください：**

```bash
# 1. コードフォーマット
uv run ruff format .

# 2. リント・品質チェック
uv run ruff check .
```

このルールは、コードの一貫性と品質を保つために必須です。

### Ruff設定

プロジェクトでは以下のruffルールを使用しています：

- **行の長さ制限**: 100文字（E501）
- **インポートの自動ソート**: isort互換（I001）
- **不要なインポートの検出**: pyflakes（F401）
- **最新のPython構文**: pyupgrade（UP015など）

### 自動修正

ほとんどのruffエラーは自動修正できます：

```bash
# 自動修正可能なエラーを修正
uv run ruff check --fix .
```

## プロジェクト構造

```
ja-complete/
├── src/
│   └── ja_complete/
│       ├── __init__.py           # パッケージエントリーポイント
│       ├── completer.py          # JaCompleterメインクラス（Facadeパターン）
│       ├── tokenizer.py          # Janomeラッパー（Singletonパターン）
│       ├── cli.py                # CLIインターフェース
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py           # 抽象基底クラス（Strategyパターン）
│       │   ├── ngram.py          # N-gram統計モデル
│       │   ├── phrase.py         # フレーズベースモデル
│       │   └── simple.py         # 単純辞書モデル
│       └── data/
│           └── default_ngram.pkl # デフォルトN-gramモデル
├── tests/                        # テストディレクトリ（src/構造と対応）
│   ├── models/
│   │   ├── test_ngram.py
│   │   ├── test_phrase.py
│   │   └── test_simple.py
│   ├── test_cli.py
│   ├── test_completer.py
│   └── test_tokenizer.py
├── scripts/
│   └── build_ngram_model.py      # N-gramモデル構築ツール
├── docs/                         # ドキュメント
├── pyproject.toml                # プロジェクト設定
├── README.md                     # ユーザー向けドキュメント
└── DEVELOPING.md                 # 開発者向けドキュメント（このファイル）
```

## 設計原則とアーキテクチャ

### デザインパターン

ja-completeは以下のGoFデザインパターンを実装しています：

1. **Facadeパターン** (`completer.py`)
   - `JaCompleter`クラスが複数のモデルとトークナイザーへの統一インターフェースを提供
   - クライアントコードから複雑なサブシステムの詳細を隠蔽

2. **Strategyパターン** (`models/base.py`)
   - `CompletionModel`抽象基底クラスが共通インターフェースを定義
   - 各補完戦略（phrase, ngram, simple）を交換可能に実装
   - Open/Closed原則に従い、新しい補完戦略を追加可能

3. **Singletonパターン** (`tokenizer.py`)
   - Janomeトークナイザーインスタンスをモジュールレベルで管理
   - 初期化コストの高いトークナイザーを1度だけ生成

### SOLID原則

- **Single Responsibility**: 各モデルクラスは1つの補完戦略のみを担当
- **Open/Closed**: 新しい補完モデルは既存コードを変更せずに追加可能
- **Liskov Substitution**: 全てのモデルは`CompletionModel`インターフェースを実装
- **Interface Segregation**: 最小限のインターフェース（`suggest`メソッドのみ）
- **Dependency Inversion**: 高レベルモジュール（JaCompleter）は抽象に依存

### ドメイン駆動設計（DDD）

プロジェクトは以下のDDD原則を適用しています：

- **境界づけられたコンテキスト**: 各補完モデルは独立したコンテキスト
- **集約**: JaCompleterが複数のモデルを集約
- **値オブジェクト**: スコア付き補完結果は不変の値オブジェクト

## 依存関係

### 実行時依存

- **janome**: 日本語形態素解析エンジン
  - トークン化、品詞タグ付け、文節抽出に使用
  - 軽量でインストールが容易

### 開発時依存

- **pytest**: テストフレームワーク
- **ruff**: 高速なリンター・フォーマッター
- **mypy**: 静的型チェック（オプション）

## N-gramモデルの構築

### カスタムモデルの作成

```bash
# 1. テキストデータを準備（.txtファイル）
mkdir training_data
echo "今日はいい天気です" > training_data/sample.txt
echo "明日は雨が降りそうです" >> training_data/sample.txt

# 2. モデルを構築
uv run python scripts/build_ngram_model.py \
  --input training_data/ \
  --output custom_model.pkl \
  --verbose

# 3. モデルを使用
from ja_complete import JaCompleter
completer = JaCompleter()
completer.load_ngram_model("custom_model.pkl")
```

### 大規模データセットでの構築

Wikipedia日本語版を使用する場合：

```bash
# 1. Wikipediaダンプをダウンロード
wget https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles.xml.bz2

# 2. WikiExtractorで平文を抽出
pip install wikiextractor
python -m wikiextractor.WikiExtractor \
  jawiki-latest-pages-articles.xml.bz2 \
  -o wiki_text/ \
  --processes 4

# 3. N-gramモデルを構築（時間がかかります）
uv run python scripts/build_ngram_model.py \
  --input wiki_text/ \
  --output wiki_ngram.pkl \
  --verbose
```

## コントリビューション

### プルリクエストの流れ

1. **ブランチを作成**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **変更を実装**
   - コードを書く
   - テストを追加
   - ドキュメントを更新

3. **コード品質チェック**
   ```bash
   uv run ruff format .
   uv run ruff check .
   uv run pytest
   ```

4. **コミット**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

5. **プッシュしてPRを作成**
   ```bash
   git push origin feature/your-feature-name
   ```

### コミットメッセージ規約

Conventional Commitsフォーマットを使用：

- `feat:` 新機能
- `fix:` バグ修正
- `docs:` ドキュメント変更
- `test:` テスト追加・修正
- `refactor:` リファクタリング
- `chore:` ビルド・設定変更

## セキュリティ

### Pickleファイルのセキュリティリスク

**重要**: N-gramモデルはPythonのpickleモジュールを使用してシリアライズされています。

**セキュリティ上の注意:**

- pickleファイルは任意のPythonコードを実行できる
- **信頼できないソースからのモデルファイルを読み込まないでください**
- カスタムモデルを読み込む際は警告が表示されます
- 信頼できるソースのみから.pklファイルを使用してください

**安全な使用方法:**

```python
# デフォルトモデル（安全）
completer = JaCompleter()

# カスタムモデル（警告が表示されます）
completer.load_ngram_model("custom_model.pkl")  # 信頼できるソースのみ！
```

**開発者向け:**

将来的には以下の改善を検討してください:
- JSON/MessagePackなどの安全なシリアライズ形式への移行
- モデルファイルの署名・検証機能
- サンドボックス化されたモデル読み込み

## トラブルシューティング

### よくある問題

**Q: テストが失敗する**
```bash
# キャッシュをクリア
uv run pytest --cache-clear

# 仮想環境を再構築
uv sync --reinstall
```

**Q: ruffエラーが解決できない**
```bash
# 自動修正を試す
uv run ruff check --fix .

# それでも解決しない場合は手動で修正
```

**Q: インポートエラーが出る**
```bash
# 開発モードでインストール
uv pip install -e .
```

## リリースプロセス

1. バージョン番号を更新（`pyproject.toml`）
2. CHANGELOGを更新
3. テストを実行: `uv run pytest`
4. タグを作成: `git tag v0.1.0`
5. PyPIにパブリッシュ: `uv publish`

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照してください。
