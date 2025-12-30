"""ja-completeのコマンドラインインターフェース。

各補完メソッドのサブコマンドを提供:
- phrase: フレーズベース補完
- ngram: N-gram統計補完
- simple: 単純辞書補完
"""

import argparse
import json
import sys
from pathlib import Path

from ja_complete import JaCompleter


def main() -> None:
    """メインCLIエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="Japanese text completion", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Completion method")

    # Phraseサブコマンド
    phrase_parser = subparsers.add_parser("phrase", help="Phrase-based completion")
    phrase_parser.add_argument("input", help="Input text")
    phrase_parser.add_argument("--phrases", help="File with phrases (one per line)")
    phrase_parser.add_argument("--top-k", type=int, default=10, help="Number of suggestions")
    phrase_parser.add_argument("--no-fallback", action="store_true", help="Disable N-gram fallback")

    # N-gramサブコマンド
    ngram_parser = subparsers.add_parser("ngram", help="N-gram completion")
    ngram_parser.add_argument("input", help="Input text")
    ngram_parser.add_argument("--model", help="Custom model path")
    ngram_parser.add_argument("--top-k", type=int, default=10, help="Number of suggestions")

    # Simpleサブコマンド
    simple_parser = subparsers.add_parser("simple", help="Simple dictionary completion")
    simple_parser.add_argument("input", help="Input text")
    simple_parser.add_argument("--dict", help="JSON file with suggestions")
    simple_parser.add_argument("--top-k", type=int, default=10, help="Number of suggestions")
    simple_parser.add_argument("--no-fallback", action="store_true", help="Disable N-gram fallback")

    args = parser.parse_args()
    completer = JaCompleter()

    try:
        # 適切なサブコマンドを実行
        if args.command == "phrase":
            # 指定されている場合はフレーズを読み込む
            if args.phrases:
                phrases_file = Path(args.phrases)
                if not phrases_file.exists():
                    print(f"Error: Phrases file not found: {args.phrases}", file=sys.stderr)
                    sys.exit(1)

                with open(phrases_file, encoding="utf-8") as f:
                    phrases = [line.strip() for line in f if line.strip()]
                completer.add_phrases(phrases)

            # 補完候補を取得
            fallback = not args.no_fallback
            results = completer.suggest_from_phrases(
                args.input, top_k=args.top_k, fallback_to_ngram=fallback
            )

        elif args.command == "ngram":
            # 指定されている場合はカスタムモデルを読み込む
            if args.model:
                completer.load_ngram_model(args.model)

            # 補完候補を取得
            results = completer.suggest_from_ngram(args.input, top_k=args.top_k)

        elif args.command == "simple":
            # 指定されている場合は辞書を読み込む
            if args.dict:
                dict_file = Path(args.dict)
                if not dict_file.exists():
                    print(f"Error: Dictionary file not found: {args.dict}", file=sys.stderr)
                    sys.exit(1)

                with open(dict_file, encoding="utf-8") as f:
                    suggestions = json.load(f)
                completer.add_simple_suggestions(suggestions)

            # 補完候補を取得
            fallback = not args.no_fallback
            results = completer.suggest_from_simple(
                args.input, top_k=args.top_k, fallback_to_ngram=fallback
            )

        else:
            parser.print_help()
            sys.exit(1)

        # 結果をJSON形式で出力
        print(json.dumps(results, ensure_ascii=False, indent=2))

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
