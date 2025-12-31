"""N-gram補完のパフォーマンスをプロファイリングする。"""

import time
from pathlib import Path

from ja_complete.models.ngram import NgramModel


def profile_ngram():
    """各ステップの実行時間を計測する。"""
    total_start = time.time()

    # 1. モデルロード
    load_start = time.time()
    model = NgramModel()
    load_time = time.time() - load_start
    print(f"Model loading: {load_time:.3f}s")

    # 2. モデル統計
    print(f"Vocabulary size: {model.vocabulary_size:,}")
    print(f"Bigrams: {len(model.bigrams):,}")
    print(f"Trigrams: {len(model.trigrams):,}")

    # 3. 補完実行
    suggest_start = time.time()
    results = model.suggest("明日の", top_k=5)
    suggest_time = time.time() - suggest_start
    print(f"\nSuggestion generation: {suggest_time:.3f}s")

    # 4. 結果表示
    print(f"\nResults ({len(results)} suggestions):")
    for i, r in enumerate(results.items, 1):
        print(f"  {i}. {r.text} (score: {r.score:.6f})")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.3f}s")

    # ファイルサイズ
    model_path = Path("src/ja_complete/data/default_ngram.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1024 / 1024
        print(f"Model file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    profile_ngram()
