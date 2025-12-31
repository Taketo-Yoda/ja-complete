"""Comprehensive tests for CLI."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from ja_complete.cli import main


class TestCLIPhrase:
    """Test phrase subcommand."""

    def test_phrase_basic(self, tmp_path, capsys):
        """Test basic phrase completion via CLI."""
        # Create phrases file
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("今日はいい天気です\n明日も晴れです\n", encoding="utf-8")

        # Mock sys.argv
        test_args = [
            "ja-complete",
            "phrase",
            "今日",
            "--phrases",
            str(phrases_file),
            "--top-k",
            "5",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        # Should output JSON
        assert captured.out
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_phrase_with_no_fallback(self, tmp_path, capsys):
        """Test phrase completion with --no-fallback flag."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("スマホを買う\n", encoding="utf-8")

        test_args = [
            "ja-complete",
            "phrase",
            "今日",  # No match
            "--phrases",
            str(phrases_file),
            "--no-fallback",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        # Should return empty list (no fallback)
        assert len(results) == 0

    def test_phrase_without_phrases_file(self, capsys):
        """Test phrase completion without phrases file."""
        test_args = ["ja-complete", "phrase", "今日", "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        # Should work but return empty or N-gram results
        assert isinstance(results, list)

    def test_phrase_file_not_found(self, capsys):
        """Test error handling when phrases file not found."""
        test_args = ["ja-complete", "phrase", "今日", "--phrases", "/nonexistent/file.txt"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "not found" in captured.err

    def test_phrase_empty_file(self, tmp_path, capsys):
        """Test phrase completion with empty phrases file."""
        phrases_file = tmp_path / "empty.txt"
        phrases_file.write_text("", encoding="utf-8")

        test_args = ["ja-complete", "phrase", "今日", "--phrases", str(phrases_file)]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_phrase_with_blank_lines(self, tmp_path, capsys):
        """Test phrase file with blank lines."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("今日はいい天気です\n\n明日も晴れです\n  \n", encoding="utf-8")

        test_args = ["ja-complete", "phrase", "今日", "--phrases", str(phrases_file)]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)


class TestCLINgram:
    """Test ngram subcommand."""

    def test_ngram_basic(self, capsys):
        """Test basic N-gram completion via CLI."""
        test_args = ["ja-complete", "ngram", "今日", "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_ngram_with_custom_model(self, tmp_path, capsys):
        """Test N-gram with custom model."""
        import pickle

        # Create minimal test model
        test_model = {
            "unigrams": {"今日": 10, "は": 8},
            "bigrams": {"今日": {"は": 5}},
            "trigrams": {},
        }

        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(test_model, f)

        test_args = ["ja-complete", "ngram", "今日", "--model", str(model_file), "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_ngram_model_file_not_found(self, capsys):
        """Test error handling when model file not found."""
        test_args = ["ja-complete", "ngram", "今日", "--model", "/nonexistent/model.pkl"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # Should exit with error
        assert exc_info.value.code == 1


class TestCLISimple:
    """Test simple subcommand."""

    def test_simple_basic(self, tmp_path, capsys):
        """Test basic simple dictionary completion via CLI."""
        # Create dictionary file
        dict_file = tmp_path / "dict.json"
        suggestions = {"お": ["おはよう", "おやすみ"], "あり": ["ありがとう"]}
        dict_file.write_text(json.dumps(suggestions, ensure_ascii=False), encoding="utf-8")

        test_args = ["ja-complete", "simple", "お", "--dict", str(dict_file), "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_simple_with_no_fallback(self, tmp_path, capsys):
        """Test simple completion with --no-fallback flag."""
        dict_file = tmp_path / "dict.json"
        suggestions = {"お": ["おはよう"]}
        dict_file.write_text(json.dumps(suggestions, ensure_ascii=False), encoding="utf-8")

        test_args = [
            "ja-complete",
            "simple",
            "あり",  # No match
            "--dict",
            str(dict_file),
            "--no-fallback",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        # Should return empty list (no fallback)
        assert len(results) == 0

    def test_simple_without_dict_file(self, capsys):
        """Test simple completion without dictionary file."""
        test_args = ["ja-complete", "simple", "test", "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        # Should work but return empty or N-gram results
        assert isinstance(results, list)

    def test_simple_dict_file_not_found(self, capsys):
        """Test error handling when dictionary file not found."""
        test_args = ["ja-complete", "simple", "test", "--dict", "/nonexistent/dict.json"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err
        assert "not found" in captured.err

    def test_simple_invalid_json(self, tmp_path, capsys):
        """Test error handling with invalid JSON file."""
        dict_file = tmp_path / "invalid.json"
        dict_file.write_text("not valid json", encoding="utf-8")

        test_args = ["ja-complete", "simple", "test", "--dict", str(dict_file)]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


class TestCLIArgumentParsing:
    """Test argument parsing."""

    def test_no_subcommand(self, capsys):
        """Test CLI with no subcommand."""
        test_args = ["ja-complete"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_subcommand(self, capsys):
        """Test CLI with invalid subcommand."""
        test_args = ["ja-complete", "invalid", "input"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_phrase_without_input(self):
        """Test phrase subcommand without input argument."""
        test_args = ["ja-complete", "phrase"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_ngram_without_input(self):
        """Test ngram subcommand without input argument."""
        test_args = ["ja-complete", "ngram"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_simple_without_input(self):
        """Test simple subcommand without input argument."""
        test_args = ["ja-complete", "simple"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                main()

    def test_top_k_argument(self, capsys):
        """Test --top-k argument parsing."""
        test_args = ["ja-complete", "ngram", "test", "--top-k", "3"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        # top_k should limit results
        assert len(results) <= 3

    def test_invalid_top_k(self, capsys):
        """Test invalid --top-k argument."""
        test_args = [
            "ja-complete",
            "ngram",
            "test",
            "--top-k",
            "0",  # Invalid
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


class TestCLIOutput:
    """Test CLI output formatting."""

    def test_output_is_valid_json(self, capsys):
        """Test that output is valid JSON."""
        test_args = ["ja-complete", "ngram", "今日"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        # Should parse as valid JSON
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_output_japanese_characters(self, tmp_path, capsys):
        """Test that Japanese characters are output correctly."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("今日はいい天気です\n", encoding="utf-8")

        test_args = ["ja-complete", "phrase", "今日", "--phrases", str(phrases_file)]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)

        # Should contain Japanese text
        if results:
            assert any("今日" in r.get("text", "") for r in results)

    def test_output_structure(self, capsys):
        """Test output structure contains text and score."""
        test_args = ["ja-complete", "ngram", "今日"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)

        for result in results:
            assert "text" in result
            assert "score" in result

    def test_output_indented(self, capsys):
        """Test that JSON output is indented."""
        test_args = ["ja-complete", "ngram", "今日"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        # Indented JSON should have newlines
        assert "\n" in captured.out


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_value_error_handling(self, capsys):
        """Test ValueError is caught and handled."""
        test_args = [
            "ja-complete",
            "phrase",
            "",  # Empty input should raise ValueError
        ]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_unexpected_error_handling(self, capsys, monkeypatch):
        """Test unexpected exception is caught and handled."""

        # Mock JaCompleter to raise unexpected error
        def mock_suggest(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        test_args = ["ja-complete", "ngram", "test"]

        with patch.object(sys, "argv", test_args):
            with patch("ja_complete.cli.JaCompleter") as mock_completer:
                mock_instance = MagicMock()
                mock_instance.suggest_from_ngram.side_effect = RuntimeError("Unexpected error")
                mock_completer.return_value = mock_instance

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1


class TestCLIIntegration:
    """Test CLI integration with JaCompleter."""

    def test_phrase_integration(self, tmp_path, capsys):
        """Test phrase subcommand integrates with JaCompleter."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text(
            "スマホの買い換えと合わせて一式揃えたい\n新生活に備えた準備を始めたい\n",
            encoding="utf-8",
        )

        test_args = ["ja-complete", "phrase", "ス", "--phrases", str(phrases_file), "--top-k", "10"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)

        # Should find phrase starting with "ス"
        assert any("スマホ" in r["text"] for r in results)

    def test_simple_integration(self, tmp_path, capsys):
        """Test simple subcommand integrates with JaCompleter."""
        dict_file = tmp_path / "dict.json"
        suggestions = {
            "お": ["おはよう", "おやすみ", "お疲れ様"],
            "あり": ["ありがとう", "ありがとうございます"],
        }
        dict_file.write_text(json.dumps(suggestions, ensure_ascii=False), encoding="utf-8")

        test_args = ["ja-complete", "simple", "あり", "--dict", str(dict_file)]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)

        # Should return both "ありがとう" suggestions
        assert len(results) == 2
        assert all("ありがとう" in r["text"] for r in results)

    def test_ngram_integration(self, capsys):
        """Test ngram subcommand integrates with JaCompleter."""
        test_args = ["ja-complete", "ngram", "今日", "--top-k", "5"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)

        # Should return results (may be empty if no default model)
        assert isinstance(results, list)


class TestCLIEdgeCases:
    """Test CLI edge cases."""

    def test_unicode_input(self, capsys):
        """Test CLI with Unicode characters."""
        test_args = ["ja-complete", "ngram", "😀今日"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_very_long_input(self, capsys):
        """Test CLI with very long input."""
        long_input = "あ" * 1000

        test_args = ["ja-complete", "ngram", long_input]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_special_characters_in_input(self, capsys):
        """Test CLI with special characters."""
        test_args = ["ja-complete", "ngram", "今日：天気"]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)

    def test_large_phrases_file(self, tmp_path, capsys):
        """Test CLI with large phrases file."""
        phrases_file = tmp_path / "large.txt"
        phrases = [f"フレーズ{i}です" for i in range(1000)]
        phrases_file.write_text("\n".join(phrases), encoding="utf-8")

        test_args = [
            "ja-complete",
            "phrase",
            "フレーズ",
            "--phrases",
            str(phrases_file),
            "--top-k",
            "10",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)
        assert len(results) <= 10

    def test_zero_results(self, tmp_path, capsys):
        """Test CLI when no results are found."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("今日はいい天気です\n", encoding="utf-8")

        test_args = [
            "ja-complete",
            "phrase",
            "xyz",  # No match
            "--phrases",
            str(phrases_file),
            "--no-fallback",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert len(results) == 0

    def test_single_result(self, tmp_path, capsys):
        """Test CLI when exactly one result is found."""
        phrases_file = tmp_path / "phrases.txt"
        phrases_file.write_text("今日はいい天気です\n", encoding="utf-8")

        test_args = [
            "ja-complete",
            "phrase",
            "今日はいい天気です",
            "--phrases",
            str(phrases_file),
            "--no-fallback",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert len(results) == 1
