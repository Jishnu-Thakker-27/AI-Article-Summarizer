"""Unit tests for src.validators module."""

from pathlib import Path
import pytest

from src.exceptions import APIKeyError, ModelNotFoundError, ValidationError
from src.validators import (
    validate_api_key,
    validate_article_text,
    validate_file_path,
    validate_model_id,
    validate_url,
)


def test_validate_url_valid() -> None:
    """Test that valid HTTP/HTTPS URLs pass validation and are normalized."""
    assert validate_url("https://example.com/article") == "https://example.com/article"
    assert validate_url("  http://sub.domain.org/path?q=1  ") == "http://sub.domain.org/path?q=1"


def test_validate_url_invalid_scheme() -> None:
    """Test that non-HTTP/HTTPS schemes raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        validate_url("ftp://example.com/file.txt")
    assert "scheme" in str(exc.value).lower()


def test_validate_url_missing_or_empty() -> None:
    """Test that empty or non-string URLs raise ValidationError."""
    with pytest.raises(ValidationError):
        validate_url("")
    with pytest.raises(ValidationError):
        validate_url(None)  # type: ignore[arg-type]


def test_validate_url_missing_netloc() -> None:
    """Test that malformed URLs missing hostnames raise ValidationError."""
    with pytest.raises(ValidationError):
        validate_url("https://")


def test_validate_article_text_valid() -> None:
    """Test that article text with sufficient words passes validation."""
    content = "This is a well-formed test article with enough words to pass the minimum validation threshold. " * 5
    result = validate_article_text(content, min_words=10)
    assert len(result.split()) >= 10


def test_validate_article_text_empty() -> None:
    """Test that empty article text raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        validate_article_text("   ")
    assert "empty" in str(exc.value).lower()


def test_validate_article_text_too_short() -> None:
    """Test that article text below minimum words raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        validate_article_text("Too short to summarize.", min_words=30)
    assert "too short" in str(exc.value).lower()


def test_validate_file_path_valid(tmp_path: Path) -> None:
    """Test that existing non-empty file passes validation."""
    sample_file = tmp_path / "article.txt"
    sample_file.write_text("Valid content inside the file.", encoding="utf-8")

    validated = validate_file_path(sample_file)
    assert validated == sample_file.resolve()


def test_validate_file_path_not_found(tmp_path: Path) -> None:
    """Test that non-existent file raises ValidationError."""
    non_existent = tmp_path / "does_not_exist.txt"
    with pytest.raises(ValidationError) as exc:
        validate_file_path(non_existent)
    assert "not found" in str(exc.value).lower()


def test_validate_file_path_empty_file(tmp_path: Path) -> None:
    """Test that zero-byte file raises ValidationError."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError) as exc:
        validate_file_path(empty_file)
    assert "empty" in str(exc.value).lower()


def test_validate_file_path_directory(tmp_path: Path) -> None:
    """Test that directory path raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        validate_file_path(tmp_path)
    assert "not a regular file" in str(exc.value).lower()


def test_validate_model_id_valid() -> None:
    """Test that valid model IDs normalize and pass."""
    catalog = {"1": {"id": "gemini-3.5-flash"}, "2": {"id": "gemini-3.5-flash-lite"}}
    assert validate_model_id("gemini-3.5-flash", catalog) == "gemini-3.5-flash"
    assert validate_model_id("models/gemini-3.5-flash", catalog) == "gemini-3.5-flash"


def test_validate_model_id_invalid() -> None:
    """Test that unrecognized model IDs raise ModelNotFoundError."""
    catalog = {"1": {"id": "gemini-3.5-flash"}}
    with pytest.raises(ModelNotFoundError):
        validate_model_id("gemini-nonexistent-model", catalog)


def test_validate_api_key_valid() -> None:
    """Test that clean API keys pass validation."""
    assert validate_api_key("AIzaSyValidGeminiKey123") == "AIzaSyValidGeminiKey123"


def test_validate_api_key_placeholder() -> None:
    """Test that obvious placeholder keys raise APIKeyError."""
    with pytest.raises(APIKeyError) as exc:
        validate_api_key("your_gemini_api_key_here")
    assert "placeholder" in str(exc.value).lower()


def test_validate_api_key_empty() -> None:
    """Test that empty or None keys raise APIKeyError."""
    with pytest.raises(APIKeyError):
        validate_api_key("")
    with pytest.raises(APIKeyError):
        validate_api_key(None)
