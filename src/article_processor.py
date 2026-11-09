"""Article processing, normalization, token estimation, and chunking for AI Article Summarizer.

Provides text cleaning, word/character counting, token estimation, and paragraph-aware
sliding-window chunking to support long articles via Map-Reduce summarization.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from src.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    MAP_REDUCE_THRESHOLD_WORDS,
    MIN_ARTICLE_WORDS,
    UNUSUALLY_LARGE_WORD_THRESHOLD,
)
from src.exceptions import ValidationError


def clean_and_normalize_text(text: str) -> str:
    """Clean and normalize input text.

    Removes zero-width characters, normalizes line endings, collapses redundant
    whitespace while preserving intentional paragraph structure.

    Args:
        text: Raw input text.

    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""

    # Remove zero-width spaces, BOM, and non-printable control characters
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Standardize line breaks to Unix newline
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # Replace runs of horizontal whitespace (spaces/tabs) with a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Ensure no more than two consecutive newlines between paragraphs
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def count_words(text: str) -> int:
    """Count words in a text string.

    Args:
        text: Input string.

    Returns:
        Total number of whitespace-delimited words.
    """
    if not text:
        return 0
    return len(text.split())


def count_characters(text: str) -> int:
    """Count total characters in a text string.

    Args:
        text: Input string.

    Returns:
        Character count.
    """
    return len(text) if text else 0


def estimate_tokens(text: str) -> int:
    """Estimate token count for Gemini models using empirical heuristic.

    Roughly ~1 word ≈ 1.33 tokens or ~4 characters per token.

    Args:
        text: Input text string.

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0
    word_count = len(text.split())
    char_count = len(text)
    # Average word-based and char-based estimates
    word_estimate = word_count * 1.33
    char_estimate = char_count / 4.0
    return max(1, int((word_estimate + char_estimate) / 2))


def analyze_article(text: str) -> Dict[str, Any]:
    """Perform comprehensive structural analysis on article text.

    Detects empty articles, short articles, and unusually large articles.

    Args:
        text: Article text.

    Returns:
        Dictionary containing counts, token estimates, and threshold flags.

    Raises:
        ValidationError: If the article is empty or below MIN_ARTICLE_WORDS.
    """
    cleaned = clean_and_normalize_text(text)
    words = count_words(cleaned)
    chars = count_characters(cleaned)
    tokens = estimate_tokens(cleaned)

    if words == 0:
        raise ValidationError(
            "Article is empty.",
            hint="Please provide article content to summarize.",
        )

    if words < MIN_ARTICLE_WORDS:
        raise ValidationError(
            f"Article is too short ({words} words). Minimum required is {MIN_ARTICLE_WORDS} words.",
            hint="Summarization requires sufficient context. Please provide a more complete article.",
        )

    is_unusually_large = words > UNUSUALLY_LARGE_WORD_THRESHOLD
    requires_map_reduce = words > MAP_REDUCE_THRESHOLD_WORDS

    return {
        "cleaned_text": cleaned,
        "word_count": words,
        "character_count": chars,
        "estimated_tokens": tokens,
        "is_unusually_large": is_unusually_large,
        "requires_map_reduce": requires_map_reduce,
    }


def chunk_text(
    text: str,
    max_chunk_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Divide long text into overlapping chunks without breaking paragraph flow.

    Uses a paragraph-first packing strategy: paragraphs are accumulated until
    reaching `max_chunk_words`. Overlap is maintained from previous chunk.
    Never silently drops or truncates any part of the article.

    Args:
        text: The normalized text to divide.
        max_chunk_words: Maximum words per chunk.
        overlap_words: Words from previous chunk to carry over for context.

    Returns:
        List of chunk strings.
    """
    cleaned = clean_and_normalize_text(text)
    total_words = count_words(cleaned)

    if total_words <= max_chunk_words:
        return [cleaned]

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: List[str] = []
    current_paragraphs: List[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = count_words(para)

        # If a single paragraph exceeds the chunk limit, split it by sentences or word blocks
        if para_words > max_chunk_words:
            # First flush what we have
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_words = 0

            # Sub-divide large paragraph
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_words = 0
            sub_chunk: List[str] = []
            for s in sentences:
                s_words = count_words(s)
                if sub_words + s_words > max_chunk_words and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_words = 0
                sub_chunk.append(s)
                sub_words += s_words
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        if current_words + para_words > max_chunk_words and current_paragraphs:
            # Emit current chunk
            chunk_str = "\n\n".join(current_paragraphs)
            chunks.append(chunk_str)

            # Build overlap from the end of the current chunk
            overlap_buffer: List[str] = []
            overlap_count = 0
            for p in reversed(current_paragraphs):
                p_w = count_words(p)
                overlap_buffer.insert(0, p)
                overlap_count += p_w
                if overlap_count >= overlap_words:
                    break

            current_paragraphs = list(overlap_buffer)
            current_words = sum(count_words(p) for p in current_paragraphs)

        current_paragraphs.append(para)
        current_words += para_words

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks
