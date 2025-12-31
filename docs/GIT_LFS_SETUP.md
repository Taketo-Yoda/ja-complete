# Git LFS セットアップガイド

このプロジェクトは、大きなモデルファイル（`*.pkl`）をGit LFS（Large File Storage）で管理するように設定されています。

## 前提条件

- Git 2.x以降がインストールされていること
- リポジトリへの書き込み権限があること

## Git LFSのインストール

### macOS (Homebrew)

```bash
brew install git-lfs
```

### Ubuntu/Debian

```bash
sudo apt-get install git-lfs
```

### Windows

1. [Git LFS公式サイト](https://git-lfs.github.com/)からインストーラーをダウンロード
2. インストーラーを実行

### インストール確認

```bash
git lfs version
# 出力例: git-lfs/3.4.0 (GitHub; darwin arm64; go 1.21.0)
```

## Git LFSの初期化

リポジトリでGit LFSを有効化します：

```bash
# リポジトリのルートディレクトリで実行
git lfs install
```

成功すると以下のメッセージが表示されます：
```
Updated Git hooks.
Git LFS initialized.
```

## 既存のファイルをGit LFSに移行

このプロジェクトでは、`.gitattributes`ファイルに既に`*.pkl`ファイルがLFS管理対象として設定されています。

既存の`default_ngram.pkl`ファイルをLFSに移行するには：

```bash
# 1. 現在のファイルをGitから削除（ワーキングディレクトリには残す）
git rm --cached src/ja_complete/data/default_ngram.pkl

# 2. LFS追跡を確認
git lfs track "*.pkl"

# 3. ファイルを再度追加（今度はLFSで追跡される）
git add src/ja_complete/data/default_ngram.pkl
git add .gitattributes

# 4. 変更をコミット
git commit -m "chore: migrate default_ngram.pkl to Git LFS"

# 5. リモートにプッシュ
git push origin main
```

## 確認方法

ファイルがGit LFSで管理されているか確認：

```bash
git lfs ls-files
```

出力例：
```
4a3b5c6d7e * src/ja_complete/data/default_ngram.pkl
```

## Git LFS使用時の注意点

### クローン時

通常通り`git clone`するだけで、LFSファイルも自動的にダウンロードされます：

```bash
git clone https://github.com/username/ja-complete.git
```

### プル時

```bash
git pull
```

LFSファイルも自動的に更新されます。

### モデルファイルのサイズ

- `default_ngram.pkl`: 7.7MB（LFS管理）
- `*_medium.pkl`: 185MB（.gitignoreで除外）
- `*_large.pkl`: それ以上（.gitignoreで除外）

中・大サイズのモデルは各自でローカルに生成してください。

## トラブルシューティング

### エラー: "This repository is configured for Git LFS but 'git-lfs' was not found"

Git LFSがインストールされていません。上記のインストール手順に従ってください。

### エラー: "Git LFS is not installed"

```bash
git lfs install
```

を実行してGit LFSを有効化してください。

### 大きなファイルのプッシュが遅い

Git LFSは大きなファイルを効率的に扱いますが、初回プッシュには時間がかかることがあります。
帯域幅に制限がある場合は、ローカルで作業し、必要な時だけプッシュすることをお勧めします。

## 参考リンク

- [Git LFS公式ドキュメント](https://git-lfs.github.com/)
- [GitHub - Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
