"""Comprehensive tests for tokenizer module."""

from ja_complete import tokenizer


class TestTokenize:
    """Test tokenize() function."""

    def test_basic_tokenization(self):
        """Test basic Japanese tokenization."""
        text = "今日はいい天気です"
        tokens = tokenizer.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_input(self):
        """Test tokenization with empty string."""
        result = tokenizer.tokenize("")
        assert result == []

    def test_whitespace_only(self):
        """Test tokenization with whitespace only."""
        result = tokenizer.tokenize("   ")
        assert result == []

    def test_hiragana_text(self):
        """Test tokenization with hiragana text."""
        text = "おはようございます"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert "おはよう" in tokens or "お" in tokens

    def test_katakana_text(self):
        """Test tokenization with katakana text."""
        text = "スマホを買う"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert "スマホ" in tokens

    def test_kanji_text(self):
        """Test tokenization with kanji text."""
        text = "新生活を始める"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert any("生活" in t for t in tokens)

    def test_mixed_script(self):
        """Test tokenization with mixed script."""
        text = "今日はいい天気ですね"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        assert "今日" in tokens
        assert "天気" in tokens

    def test_particles(self):
        """Test tokenization correctly identifies particles."""
        text = "私は学生です"
        tokens = tokenizer.tokenize(text)
        assert "は" in tokens  # Particle
        assert "私" in tokens  # Noun

    def test_complex_sentence(self):
        """Test tokenization with complex sentence."""
        text = "スマホの買い換えと合わせて一式揃えたい"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 5
        # Janome might tokenize "スマホ" as separate morphemes
        assert "買い換え" in tokens or "買い" in tokens

    def test_numeric_text(self):
        """Test tokenization with numbers."""
        text = "今日は12月30日です"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        # Numbers might be kept as-is or tokenized

    def test_punctuation(self):
        """Test tokenization with punctuation."""
        text = "こんにちは、元気ですか？"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0
        # Janome may or may not include punctuation


class TestGetMorphemes:
    """Test get_morphemes() function."""

    def test_basic_morphemes(self):
        """Test basic morphological analysis."""
        text = "今日は晴れです"
        morphemes = tokenizer.get_morphemes(text)
        assert isinstance(morphemes, list)
        assert len(morphemes) > 0

    def test_morpheme_structure(self):
        """Test morpheme dictionary structure."""
        text = "私は学生です"
        morphemes = tokenizer.get_morphemes(text)

        for morpheme in morphemes:
            assert isinstance(morpheme, dict)
            assert "surface" in morpheme
            assert "base_form" in morpheme
            assert "pos" in morpheme  # 'pos' not 'part_of_speech'

    def test_empty_text_morphemes(self):
        """Test get_morphemes with empty string."""
        result = tokenizer.get_morphemes("")
        assert result == []

    def test_particle_identification(self):
        """Test identification of particles in morphemes."""
        text = "私は学生です"
        morphemes = tokenizer.get_morphemes(text)

        # Find particle "は"
        particle = next((m for m in morphemes if m["surface"] == "は"), None)
        assert particle is not None
        assert particle["pos"] == "助詞"

    def test_noun_identification(self):
        """Test identification of nouns in morphemes."""
        text = "今日は天気です"
        morphemes = tokenizer.get_morphemes(text)

        # Find noun "今日" or "天気"
        nouns = [m for m in morphemes if m["pos"] == "名詞"]
        assert len(nouns) > 0

    def test_verb_identification(self):
        """Test identification of verbs in morphemes."""
        text = "走る"
        morphemes = tokenizer.get_morphemes(text)

        verb = next((m for m in morphemes if m["pos"] == "動詞"), None)
        assert verb is not None

    def test_base_form_extraction(self):
        """Test base form extraction."""
        text = "走った"  # Past tense of 走る
        morphemes = tokenizer.get_morphemes(text)

        # Base form should be 走る
        verb_morpheme = next((m for m in morphemes if m["pos"] == "動詞"), None)
        if verb_morpheme:
            assert verb_morpheme["base_form"] == "走る"

    def test_complex_sentence_morphemes(self):
        """Test morphemes for complex sentence."""
        text = "スマホを買い換える"
        morphemes = tokenizer.get_morphemes(text)

        assert len(morphemes) >= 3
        # Should have noun (スマホ), particle (を), and verb components


class TestExtractBunsetsu:
    """Test extract_bunsetsu() function."""

    def test_basic_bunsetsu_extraction(self):
        """Test basic bunsetsu extraction."""
        text = "今日は晴れです"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        assert isinstance(bunsetsu, list)
        assert len(bunsetsu) > 0
        assert all(isinstance(b, str) for b in bunsetsu)

    def test_empty_text_bunsetsu(self):
        """Test bunsetsu extraction with empty string."""
        result = tokenizer.extract_bunsetsu("")
        assert result == []

    def test_particle_boundary(self):
        """Test bunsetsu breaks after particles."""
        text = "私は学生です"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        # Should break after particle は
        # Expected: ["私は", "学生です"] or similar
        assert len(bunsetsu) >= 2

    def test_complex_bunsetsu(self):
        """Test bunsetsu with complex sentence."""
        text = "スマホの買い換えと合わせて一式揃えたい"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        # Should have multiple bunsetsu
        assert len(bunsetsu) >= 2
        # Joined bunsetsu should reconstruct original (minus spaces)
        joined = "".join(bunsetsu)
        assert joined == text

    def test_bunsetsu_reconstruction(self):
        """Test that bunsetsu can be joined to reconstruct original."""
        text = "今日はいい天気ですね"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        joined = "".join(bunsetsu)
        assert joined == text

    def test_single_bunsetsu(self):
        """Test text that forms single bunsetsu."""
        text = "こんにちは"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        # Should be at least one bunsetsu
        assert len(bunsetsu) >= 1

    def test_multiple_particles(self):
        """Test bunsetsu with multiple particles."""
        text = "私はあなたに本をあげます"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        # Should break at particles は, に, を
        assert len(bunsetsu) >= 3

    def test_noun_verb_bunsetsu(self):
        """Test bunsetsu with noun-verb structure."""
        text = "犬が走る"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        assert len(bunsetsu) >= 1
        joined = "".join(bunsetsu)
        assert joined == text

    def test_long_sentence_bunsetsu(self):
        """Test bunsetsu extraction for long sentence."""
        text = "新生活に備えた準備を始めたい"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        assert len(bunsetsu) >= 2
        joined = "".join(bunsetsu)
        assert joined == text

    def test_bunsetsu_non_empty_segments(self):
        """Test that all bunsetsu segments are non-empty."""
        text = "今日はいい天気です"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        assert all(len(b) > 0 for b in bunsetsu)

    def test_bunsetsu_ordering(self):
        """Test that bunsetsu maintains original order."""
        text = "春が来た"
        bunsetsu = tokenizer.extract_bunsetsu(text)

        # First bunsetsu should contain 春
        assert "春" in bunsetsu[0]

        # Reconstruction should match original
        assert "".join(bunsetsu) == text


class TestTokenizerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_long_text(self):
        """Test tokenization with very long text."""
        text = "今日はいい天気です。" * 100
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0

    def test_special_characters(self):
        """Test tokenization with special characters."""
        text = "メール：test@example.com"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0

    def test_mixed_japanese_english(self):
        """Test tokenization with mixed Japanese and English."""
        text = "Pythonで開発する"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0

    def test_repeated_characters(self):
        """Test tokenization with repeated characters."""
        text = "ああああああ"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) > 0

    def test_single_character(self):
        """Test tokenization with single character."""
        text = "私"
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 1
        assert tokens[0] == "私"

    def test_tokenize_consistency(self):
        """Test that tokenize returns consistent results."""
        text = "今日は晴れです"
        result1 = tokenizer.tokenize(text)
        result2 = tokenizer.tokenize(text)
        assert result1 == result2

    def test_morphemes_consistency(self):
        """Test that get_morphemes returns consistent results."""
        text = "今日は晴れです"
        result1 = tokenizer.get_morphemes(text)
        result2 = tokenizer.get_morphemes(text)
        assert result1 == result2

    def test_bunsetsu_consistency(self):
        """Test that extract_bunsetsu returns consistent results."""
        text = "今日は晴れです"
        result1 = tokenizer.extract_bunsetsu(text)
        result2 = tokenizer.extract_bunsetsu(text)
        assert result1 == result2
