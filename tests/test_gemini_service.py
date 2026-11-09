"""Unit tests for src.gemini_service module."""

from unittest.mock import MagicMock
import pytest
from google.genai import errors

from src.exceptions import (
    APIKeyError,
    ModelNotFoundError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    SummarizerException,
)
from src.gemini_service import (
    create_gemini_client,
    generate_summary,
    generate_summary_stream,
    map_api_exception,
)
from src.performance import PerformanceTracker
from src.prompt_templates import parse_structured_output


def test_create_gemini_client_invalid_key() -> None:
    """Test that missing or placeholder keys raise APIKeyError."""
    with pytest.raises(APIKeyError):
        create_gemini_client(api_key="your_gemini_api_key_here")
    with pytest.raises(APIKeyError):
        create_gemini_client(api_key="")


def test_map_api_exception_auth_errors() -> None:
    """Test mapping HTTP 401 and 403 to APIKeyError."""
    err_401 = errors.ClientError(401, {"error": {"code": 401, "message": "API_KEY_INVALID"}}, None)
    mapped = map_api_exception(err_401)
    assert isinstance(mapped, APIKeyError)

    err_403 = errors.ClientError(403, {"error": {"code": 403, "message": "Permission denied"}}, None)
    mapped_403 = map_api_exception(err_403)
    assert isinstance(mapped_403, APIKeyError)


def test_map_api_exception_model_not_found() -> None:
    """Test mapping HTTP 404 to ModelNotFoundError."""
    err_404 = errors.ClientError(404, {"error": {"code": 404, "message": "Model not found"}}, None)
    mapped = map_api_exception(err_404)
    assert isinstance(mapped, ModelNotFoundError)


def test_map_api_exception_rate_limits() -> None:
    """Test mapping HTTP 429 to RateLimitError and QuotaExceededError."""
    err_429_rpm = errors.ClientError(429, {"error": {"code": 429, "message": "Rate limit exceeded"}}, None)
    assert isinstance(map_api_exception(err_429_rpm), RateLimitError)

    err_429_quota = errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded for quota metric"}}, None)
    assert isinstance(map_api_exception(err_429_quota), QuotaExceededError)


def test_map_api_exception_server_errors() -> None:
    """Test mapping HTTP 503 to temporary service unavailability."""
    err_503 = errors.ServerError(503, {"error": {"code": 503, "message": "Overloaded"}}, None)
    mapped = map_api_exception(err_503)
    assert isinstance(mapped, SummarizerException)
    assert "temporarily unavailable" in mapped.message


def test_map_api_exception_timeout() -> None:
    """Test mapping timeout errors to NetworkError."""
    mapped = map_api_exception(TimeoutError("Connection timed out"))
    assert isinstance(mapped, NetworkError)


def test_generate_summary_mock() -> None:
    """Test synchronous generation with mock Gemini client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a concise generated summary."
    mock_response.candidates = [MagicMock(finish_reason="STOP")]
    mock_response.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50, total_token_count=150)
    mock_client.models.generate_content.return_value = mock_response

    result = generate_summary(
        client=mock_client,
        prompt="Summarize this text",
        model_id="gemini-3.5-flash",
    )

    assert result["text"] == "This is a concise generated summary."
    assert result["usage_metadata"].total_token_count == 150


def test_generate_summary_stream_mock() -> None:
    """Test streaming generation callback and performance tracking."""
    mock_client = MagicMock()
    
    # Simulate three streamed chunks
    chunk1 = MagicMock()
    chunk1.text = "Chunk 1 "
    chunk1.candidates = [MagicMock(finish_reason="STOP")]
    chunk1.usage_metadata = None

    chunk2 = MagicMock()
    chunk2.text = "Chunk 2 "
    chunk2.candidates = [MagicMock(finish_reason="STOP")]
    chunk2.usage_metadata = None

    chunk3 = MagicMock()
    chunk3.text = "Chunk 3"
    chunk3.candidates = [MagicMock(finish_reason="STOP")]
    chunk3.usage_metadata = MagicMock(prompt_token_count=50, candidates_token_count=30, total_token_count=80)

    mock_client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]

    chunks_received = []
    tracker = PerformanceTracker(input_word_count=200)

    result = generate_summary_stream(
        client=mock_client,
        prompt="Summarize",
        model_id="gemini-3.5-flash",
        on_chunk=lambda c: chunks_received.append(c),
        performance_tracker=tracker,
    )

    assert result["text"] == "Chunk 1 Chunk 2 Chunk 3"
    assert chunks_received == ["Chunk 1 ", "Chunk 2 ", "Chunk 3"]
    assert tracker.metrics.total_tokens == 80
    assert tracker.metrics.first_chunk_latency_sec is not None


def test_parse_structured_output_json() -> None:
    """Test parsing JSON-formatted Gemini response."""
    raw_json = """```json
    {
        "title": "Quantum Leap",
        "tldr": "A major quantum computing milestone.",
        "summary": "Full summary text here.",
        "key_takeaways": ["Takeaway 1", "Takeaway 2", "Takeaway 3"],
        "topics_and_keywords": ["Physics", "Computing"],
        "central_message": "Quantum supremacy is near."
    }
    ```"""
    parsed = parse_structured_output(raw_json)
    assert parsed["title"] == "Quantum Leap"
    assert len(parsed["key_takeaways"]) == 3
    assert parsed["tldr"] == "A major quantum computing milestone."


def test_parse_structured_output_markdown() -> None:
    """Test parsing markdown formatted Gemini response."""
    md_text = """# Title: Renewable Energy Revolution

## TL;DR
Solar and wind power are overtaking fossil fuels.

## Summary
The global transition toward renewable energy has accelerated dramatically over the last decade.

## Key Takeaways
- Solar panel production costs have declined by 80%.
- Grid-scale battery storage is stabilizing intermittent supply.
- Policy mandates are accelerating enterprise adoption.

## Important Topics & Keywords
Solar, Wind, Battery Storage, Energy Transition

## Central Message
Renewable power is now cheaper and more reliable than coal.
"""
    parsed = parse_structured_output(md_text)
    assert parsed["title"] == "Renewable Energy Revolution"
    assert "Solar and wind power" in parsed["tldr"]
    assert len(parsed["key_takeaways"]) == 3
    assert "Solar" in parsed["topics_and_keywords"]
    assert "Renewable power is now cheaper" in parsed["central_message"]
