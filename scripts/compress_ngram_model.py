"""N-gramモデルを圧縮して、ファイルサイズを削減する。

低頻度のN-gramを削除することで、品質を大きく損なうことなく
ファイルサイズを大幅に削減する。
"""

import argparse
import pickle
from pathlib import Path


def compress_model(
    input_path: Path,
    output_path: Path,
    min_unigram_count: int = 3,
    min_bigram_count: int = 2,
    min_trigram_count: int = 2,
    verbose: bool = False,
) -> None:
    """
    N-gramモデルを圧縮する。

    Args:
        input_path: 入力モデルファイルのパス
        output_path: 出力モデルファイルのパス
        min_unigram_count: ユニグラムの最小出現回数
        min_bigram_count: バイグラムの最小出現回数
        min_trigram_count: トライグラムの最小出現回数
        verbose: 詳細な情報を出力
    """
    # モデルを読み込む
    if verbose:
        print(f"Loading model from {input_path}...")
    with open(input_path, "rb") as f:
        model = pickle.load(f)

    unigrams = model.get("unigrams", {})
    bigrams = model.get("bigrams", {})
    trigrams = model.get("trigrams", {})

    if verbose:
        print(f"Original model statistics:")
        print(f"  Unigrams: {len(unigrams):,}")
        print(f"  Bigrams: {len(bigrams):,}")
        print(f"  Trigrams: {len(trigrams):,}")

    # ユニグラムをフィルタリング
    filtered_unigrams = {
        token: count for token, count in unigrams.items() if count >= min_unigram_count
    }

    # バイグラムをフィルタリング
    filtered_bigrams = {}
    for context, next_tokens in bigrams.items():
        # ユニグラムに存在するトークンのみを保持
        if context in filtered_unigrams:
            filtered_next = {
                token: count
                for token, count in next_tokens.items()
                if count >= min_bigram_count and token in filtered_unigrams
            }
            if filtered_next:
                filtered_bigrams[context] = filtered_next

    # トライグラムをフィルタリング
    filtered_trigrams = {}
    for context, next_tokens in trigrams.items():
        # 両方のコンテキストトークンがユニグラムに存在する場合のみ保持
        if context[0] in filtered_unigrams and context[1] in filtered_unigrams:
            filtered_next = {
                token: count
                for token, count in next_tokens.items()
                if count >= min_trigram_count and token in filtered_unigrams
            }
            if filtered_next:
                filtered_trigrams[context] = filtered_next

    if verbose:
        print(f"\nFiltered model statistics:")
        print(f"  Unigrams: {len(filtered_unigrams):,} ({len(filtered_unigrams)/len(unigrams)*100:.1f}%)")
        print(f"  Bigrams: {len(filtered_bigrams):,} ({len(filtered_bigrams)/len(bigrams)*100:.1f}% if bigrams else 0)")
        print(f"  Trigrams: {len(filtered_trigrams):,} ({len(filtered_trigrams)/len(trigrams)*100:.1f}% if trigrams else 0)")

    # 圧縮されたモデルを保存
    compressed_model = {
        "unigrams": filtered_unigrams,
        "bigrams": filtered_bigrams,
        "trigrams": filtered_trigrams,
    }

    if verbose:
        print(f"\nSaving compressed model to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(compressed_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ファイルサイズを比較
    original_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    compression_ratio = (1 - compressed_size / original_size) * 100

    if verbose:
        print(f"\nCompression results:")
        print(f"  Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"  Compressed size: {compressed_size / 1024 / 1024:.1f} MB")
        print(f"  Reduction: {compression_ratio:.1f}%")


def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="N-gramモデルを圧縮してファイルサイズを削減する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で圧縮
  python scripts/compress_ngram_model.py \\
    --input src/ja_complete/data/default_ngram.pkl \\
    --output src/ja_complete/data/default_ngram_compressed.pkl

  # より積極的な圧縮（品質が若干低下）
  python scripts/compress_ngram_model.py \\
    --input src/ja_complete/data/default_ngram.pkl \\
    --output src/ja_complete/data/default_ngram_compressed.pkl \\
    --min-unigram 5 --min-bigram 3 --min-trigram 3
        """,
    )

    parser.add_argument("--input", required=True, help="入力モデルファイル（.pkl）のパス")
    parser.add_argument("--output", required=True, help="出力モデルファイル（.pkl）のパス")
    parser.add_argument(
        "--min-unigram",
        type=int,
        default=3,
        help="ユニグラムの最小出現回数（デフォルト: 3）",
    )
    parser.add_argument(
        "--min-bigram",
        type=int,
        default=2,
        help="バイグラムの最小出現回数（デフォルト: 2）",
    )
    parser.add_argument(
        "--min-trigram",
        type=int,
        default=2,
        help="トライグラムの最小出現回数（デフォルト: 2）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細な情報を出力")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    output_path = Path(args.output)

    compress_model(
        input_path,
        output_path,
        args.min_unigram,
        args.min_bigram,
        args.min_trigram,
        args.verbose,
    )


if __name__ == "__main__":
    main()
