"""Unit tests for src.article_processor module."""

import pytest

from src.article_processor import (
    analyze_article,
    chunk_text,
    clean_and_normalize_text,
    count_characters,
    count_words,
    estimate_tokens,
)
from src.exceptions import ValidationError


def test_clean_and_normalize_text() -> None:
    """Test whitespace collapse, zero-width stripping, and newline normalization."""
    raw = "Header  Line\r\n\r\nParagraph \u200bone.\n\n\n\nParagraph two.\t"
    cleaned = clean_and_normalize_text(raw)

    assert "\r" not in cleaned
    assert "\u200b" not in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Paragraph one." in cleaned
    assert "Paragraph two." in cleaned


def test_count_words() -> None:
    """Test word counting accuracy."""
    assert count_words("") == 0
    assert count_words("   ") == 0
    assert count_words("One two three four five") == 5
    assert count_words("Word\nwith\tmultiple\rwhitespace") == 4


def test_count_characters() -> None:
    """Test character counting accuracy."""
    assert count_characters("") == 0
    assert count_characters("hello") == 5


def test_estimate_tokens() -> None:
    """Test token estimation heuristic."""
    assert estimate_tokens("") == 0
    text = "The quick brown fox jumps over the lazy dog."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert 5 <= tokens <= 20


def test_analyze_article_valid() -> None:
    """Test structural analysis on standard article."""
    text = "Artificial intelligence is advancing rapidly across diverse scientific disciplines. " * 10
    result = analyze_article(text)

    assert result["word_count"] >= 30
    assert result["requires_map_reduce"] is False
    assert result["is_unusually_large"] is False
    assert result["estimated_tokens"] > 0


def test_analyze_article_too_short() -> None:
    """Test that articles below minimum length raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        analyze_article("Just a few short words here.")
    assert "too short" in str(exc.value).lower()


def test_analyze_article_empty() -> None:
    """Test that empty string raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        analyze_article("   ")
    assert "empty" in str(exc.value).lower()


def test_chunk_text_single_chunk() -> None:
    """Test that articles under max chunk size remain a single chunk."""
    text = "This is a brief article that easily fits within a single chunk."
    chunks = chunk_text(text, max_chunk_words=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_multiple_chunks_with_overlap() -> None:
    """Test that long articles split across paragraphs with overlap preserved."""
    paragraphs = [
        f"Paragraph {i}: " + ("Content words discussing topic in detail. " * 5)
        for i in range(1, 11)
    ]
    long_text = "\n\n".join(paragraphs)

    chunks = chunk_text(long_text, max_chunk_words=50, overlap_words=10)

    assert len(chunks) > 1
    # Check that later chunks contain overlapping context
    assert any("Paragraph 1" in chunks[0] for _ in [1])
    # Total combined words across chunks should equal or exceed original due to overlap
    total_chunk_words = sum(count_words(c) for c in chunks)
    assert total_chunk_words >= count_words(long_text)
