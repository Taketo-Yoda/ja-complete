# Contributing to ja-complete

ja-completeへの貢献に興味を持っていただきありがとうございます！

## 開発環境のセットアップ

### 必要要件

- Python 3.10以上
- [uv](https://docs.astral.sh/uv/) (推奨パッケージマネージャー)

### セットアップ手順

```bash
# リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/ja-complete.git
cd ja-complete

# 依存関係のインストール
uv sync

# 開発用の追加依存関係も含める場合
uv sync --all-extras
```

## 開発ワークフロー

### ブランチ戦略

- `main`: 安定版（リリース済み）。常にデプロイ可能な状態
- `develop`: 開発版。次のリリースに向けた開発
- `feature/*`: 機能開発用（`develop`から分岐、`develop`にマージ）
- `fix/*`: バグ修正用（`develop`から分岐、`develop`にマージ）

### 開発の流れ

1. **developブランチから新しいブランチを作成**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **コードを書く**
   - コードを実装
   - 必要に応じてテストを追加

3. **テストとコード品質チェック**
   ```bash
   # テストの実行
   uv run pytest

   # Lintチェック
   uv run ruff check src/ tests/

   # 型チェック（オプション）
   uv run mypy src/
   ```

4. **コミット**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```

   コミットメッセージの形式:
   - `feat:` 新機能
   - `fix:` バグ修正
   - `docs:` ドキュメントのみの変更
   - `test:` テストの追加・修正
   - `refactor:` リファクタリング
   - `chore:` その他の変更

5. **プッシュとPull Request作成**
   ```bash
   git push origin feature/your-feature-name
   ```

   GitHubでPull Requestを作成し、`develop`ブランチへのマージをリクエスト

## テスト

### 全テストの実行

```bash
uv run pytest
```

### 特定のテストファイルのみ実行

```bash
uv run pytest tests/test_completer.py
```

### 特定のテストクラス・メソッドのみ実行

```bash
uv run pytest tests/test_completer.py::TestJaCompleter::test_initialization
```

### カバレッジレポート付きで実行

```bash
uv run pytest --cov=ja_complete --cov-report=html
```

## コード品質

### Lintチェック

```bash
# チェックのみ
uv run ruff check src/ tests/

# 自動修正
uv run ruff check --fix src/ tests/
```

### フォーマット

```bash
# チェックのみ
uv run ruff format --check src/ tests/

# フォーマット実行
uv run ruff format src/ tests/
```

## リリースプロセス

メンテナー向けの手順です。

### 1. バージョンの更新

`src/ja_complete/__init__.py` の `__version__` を更新:

```python
__version__ = "0.x.x"
```

### 2. テストとチェック

```bash
# 全テストの実行
uv run pytest

# Lintチェック
uv run ruff check src/ tests/

# ビルドのテスト
uv build
```

### 3. 変更のコミット

```bash
git add src/ja_complete/__init__.py
git commit -m "chore: bump version to 0.x.x"
git push origin develop
```

### 4. developをmainにマージ

```bash
git checkout main
git merge develop
git push origin main
```

### 5. GitHubタグの作成

```bash
git tag -a v0.x.x -m "Release v0.x.x

- 主な変更点1
- 主な変更点2
- 主な変更点3
"

git push origin v0.x.x
```

### 6. PyPIへの公開

```bash
# TestPyPIでテスト（推奨）
uv publish --repository testpypi

# 本番PyPIへ公開
uv publish
```

### 7. GitHubリリースの作成

1. GitHubのリポジトリページで "Releases" → "Create a new release"
2. タグ `v0.x.x` を選択
3. リリースノートを記載
4. "Publish release"をクリック

## Pull Requestのガイドライン

### PRを作成する前に

- [ ] 全テストがパスすることを確認
- [ ] Lintチェックがパスすることを確認
- [ ] 新機能の場合、適切なテストを追加
- [ ] ドキュメント（README.md等）を更新（必要に応じて）

### PRの説明

PRには以下を含めてください:

- **変更の概要**: 何を変更したか
- **変更の理由**: なぜこの変更が必要か
- **テスト**: どのようにテストしたか
- **関連Issue**: あれば記載

### レビュープロセス

- PRはメンテナーによってレビューされます
- フィードバックに対応してください
- 承認後、`develop`ブランチにマージされます

## コーディング規約

### Pythonスタイル

- PEP 8に従う
- Ruffによる自動フォーマット・Lintを使用
- 型ヒントを可能な限り使用

### ドキュメント

- すべての公開関数・クラスにdocstringを記載
- docstringはGoogle形式を使用
- 例を含める（可能な限り）

### テスト

- 新機能には必ずテストを追加
- テストは具体的で理解しやすく
- テストメソッド名は `test_` で始める
- テストクラス名は `Test` で始める

## 質問・相談

- **Issue**: バグ報告や機能要望は[GitHub Issues](https://github.com/YOUR_USERNAME/ja-complete/issues)
- **Discussion**: 質問や議論は[GitHub Discussions](https://github.com/YOUR_USERNAME/ja-complete/discussions)

## ライセンス

貢献されたコードは、プロジェクトと同じライセンス（MIT License）の下で公開されます。
