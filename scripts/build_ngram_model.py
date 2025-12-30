"""テキストファイルからN-gramモデルを構築する。

このスクリプトはテキストファイルを処理し、統計的補完のための
バイグラムおよびトライグラムモデルを構築する。
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

# ja_completeからトークナイザーをインポート
# 注: ja_completeがインストールされているか、PYTHONPATHが設定されている必要がある
try:
    from ja_complete import tokenizer
except ImportError:
    import sys

    # 開発用にsrcディレクトリをパスに追加
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from ja_complete import tokenizer


def build_ngram_model(input_dir: Path, output_path: Path, verbose: bool = False) -> None:
    """
    テキストファイルからN-gramモデルを構築する。

    Args:
        input_dir: テキストファイル（.txt）を含むディレクトリ
        output_path: pickle化されたモデルを保存するパス
        verbose: 進行状況の情報を出力する
    """
    unigrams = Counter()
    bigrams = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    # input_dir内の全ての.txtファイルを処理（再帰的）
    text_files = list(input_dir.rglob("*.txt"))

    if not text_files:
        print(f"Warning: No .txt files found in {input_dir}")
        return

    if verbose:
        print(f"Found {len(text_files)} text files")

    processed_lines = 0
    for text_file in text_files:
        if verbose:
            print(f"Processing: {text_file}")

        with open(text_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 行をトークン化
                tokens = tokenizer.tokenize(line)
                if not tokens:
                    continue

                # ユニグラムをカウント
                unigrams.update(tokens)

                # バイグラムをカウント
                for i in range(len(tokens) - 1):
                    bigrams[tokens[i]][tokens[i + 1]] += 1

                # トライグラムをカウント
                for i in range(len(tokens) - 2):
                    trigrams[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

                processed_lines += 1

                if verbose and processed_lines % 1000 == 0:
                    print(f"Processed {processed_lines} lines...")

    if verbose:
        print(f"Total lines processed: {processed_lines}")
        print(f"Vocabulary size: {len(unigrams)}")
        print(f"Bigrams: {len(bigrams)}")
        print(f"Trigrams: {len(trigrams)}")

    # pickle化のためにdefaultdictを通常のdictに変換
    model = {
        "unigrams": dict(unigrams),
        "bigrams": {k: dict(v) for k, v in bigrams.items()},
        "trigrams": {k: dict(v) for k, v in trigrams.items()},
    }

    # モデルを保存
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    if verbose:
        print(f"Model saved to {output_path}")


def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="テキストファイルからN-gramモデルを構築する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # テキストファイルのディレクトリからモデルを構築
  python scripts/build_ngram_model.py --input wiki_text/ --output model.pkl

  # 詳細な出力を表示
  python scripts/build_ngram_model.py --input wiki_text/ --output model.pkl --verbose
        """,
    )

    parser.add_argument(
        "--input", required=True, help="テキストファイル（.txt）を含む入力ディレクトリ"
    )
    parser.add_argument("--output", required=True, help="pickle化されたモデル（.pkl）の出力パス")
    parser.add_argument("--verbose", "-v", action="store_true", help="進行状況の情報を出力する")

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {args.input}")
        return

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {args.input}")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    build_ngram_model(input_dir, output_path, args.verbose)


if __name__ == "__main__":
    main()
