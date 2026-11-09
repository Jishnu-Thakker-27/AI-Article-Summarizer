"""Input validation routines for AI Article Summarizer.

Provides robust validation for URLs, file paths, article text content,
model identifiers, and API keys with clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from src.config import (
    MIN_ARTICLE_WORDS,
    UNUSUALLY_LARGE_WORD_THRESHOLD,
)
from src.exceptions import APIKeyError, ModelNotFoundError, ValidationError


def validate_url(url: str | None) -> str:
    """Validate that the provided string is a valid HTTP or HTTPS URL.

    Args:
        url: The candidate URL string.

    Returns:
        The validated and stripped URL string.

    Raises:
        ValidationError: If the URL is empty, missing scheme, or not http/https.
    """
    if not url or not isinstance(url, str):
        raise ValidationError(
            "URL cannot be empty.",
            hint="Please provide a valid URL starting with http:// or https://.",
        )

    clean_url = url.strip()
    try:
        parsed = urlparse(clean_url)
    except Exception as exc:
        raise ValidationError(
            f"Failed to parse URL '{clean_url}': {exc}",
            hint="Double check the URL formatting.",
        ) from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise ValidationError(
            f"Invalid URL scheme '{parsed.scheme}'. Only 'http' and 'https' are supported.",
            hint="Ensure your URL begins with http:// or https://.",
        )

    if not parsed.netloc:
        raise ValidationError(
            f"URL '{clean_url}' is missing a valid domain name.",
            hint="Example: https://example.com/article",
        )

    return clean_url


def validate_article_text(
    text: str | None,
    min_words: int = MIN_ARTICLE_WORDS,
    max_words: int = UNUSUALLY_LARGE_WORD_THRESHOLD,
) -> str:
    """Validate article content for non-emptiness and sensible word counts.

    Args:
        text: Raw article text.
        min_words: Minimum acceptable word count.
        max_words: Warning threshold for unusually large inputs.

    Returns:
        The cleaned, non-empty article text.

    Raises:
        ValidationError: If the article text is empty or too short.
    """
    if text is None:
        raise ValidationError(
            "Article content is missing.",
            hint="Please paste or supply article text.",
        )

    clean_text = text.strip()
    if not clean_text:
        raise ValidationError(
            "Article text is empty.",
            hint="Provide text with substantive content to summarize.",
        )

    words = clean_text.split()
    word_count = len(words)

    if word_count < min_words:
        raise ValidationError(
            f"Article is too short ({word_count} words). Minimum required is {min_words} words.",
            hint="Summarization requires sufficient context. Please provide a more complete article.",
        )

    return clean_text


def validate_file_path(path: Path | str | None) -> Path:
    """Validate that a local file path exists, is a regular file, and is readable.

    Args:
        path: Path to the target text file.

    Returns:
        The resolved Path object.

    Raises:
        ValidationError: If path is missing, does not exist, is a directory, or is empty.
    """
    if not path:
        raise ValidationError(
            "File path cannot be empty.",
            hint="Specify a valid path to a text file (e.g., --input-file article.txt).",
        )

    resolved = Path(path).resolve()

    if not resolved.exists():
        raise ValidationError(
            f"Input file not found: {resolved}",
            hint="Check that the file path is correct and exists on disk.",
        )

    if not resolved.is_file():
        raise ValidationError(
            f"Path is not a regular file: {resolved}",
            hint="Please point to a text file, not a directory.",
        )

    try:
        size = resolved.stat().st_size
        if size == 0:
            raise ValidationError(
                f"File '{resolved.name}' is empty (0 bytes).",
                hint="Provide a file that contains article text.",
            )
    except OSError as exc:
        raise ValidationError(
            f"Cannot inspect file '{resolved.name}': {exc}",
            hint="Check file permissions.",
        ) from exc

    return resolved


def validate_model_id(
    model_id: str | None,
    available_models: dict | list | None = None,
) -> str:
    """Validate a requested Gemini model ID against known or available models.

    Args:
        model_id: Model identifier (e.g. 'gemini-3.5-flash').
        available_models: Optional catalog or list of model IDs to validate against.

    Returns:
        The validated model ID string (without 'models/' prefix).

    Raises:
        ModelNotFoundError: If the model ID is unrecognized.
    """
    if not model_id or not isinstance(model_id, str):
        raise ModelNotFoundError(
            str(model_id),
            hint="Provide a valid model ID, such as 'gemini-3.5-flash'.",
        )

    clean_id = model_id.strip()
    if clean_id.startswith("models/"):
        clean_id = clean_id.replace("models/", "")

    if available_models:
        known_ids: set[str] = set()
        if isinstance(available_models, dict):
            for v in available_models.values():
                if isinstance(v, dict) and "id" in v:
                    known_ids.add(v["id"].replace("models/", ""))
                elif isinstance(v, str):
                    known_ids.add(v.replace("models/", ""))
        elif isinstance(available_models, (list, set, tuple)):
            for item in available_models:
                if isinstance(item, dict) and "id" in item:
                    known_ids.add(item["id"].replace("models/", ""))
                elif isinstance(item, str):
                    known_ids.add(item.replace("models/", ""))

        if known_ids and clean_id not in known_ids:
            known_list = ", ".join(sorted(known_ids))
            raise ModelNotFoundError(
                clean_id,
                hint=f"Available models are: {known_list}",
            )

    return clean_id


def validate_api_key(api_key: str | None) -> str:
    """Validate the Gemini API key.

    Args:
        api_key: The candidate API key string.

    Returns:
        The validated API key string.

    Raises:
        APIKeyError: If the key is missing or is an obvious placeholder.
    """
    if not api_key or not isinstance(api_key, str) or not api_key.strip():
        raise APIKeyError()

    clean_key = api_key.strip()
    placeholder_signatures = [
        "your_gemini_api_key",
        "your_api_key_here",
        "placeholder",
        "enter_key_here",
    ]
    if any(p in clean_key.lower() for p in placeholder_signatures):
        raise APIKeyError(
            message="Gemini API key is currently set to a placeholder value.",
            hint="Replace the placeholder in your .env file with your actual Google Gemini API key.",
        )

    return clean_key
