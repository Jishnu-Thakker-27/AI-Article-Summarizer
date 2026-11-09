"""Tests for FastAPI backend endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


@pytest.fixture
def client():
    """Create a FastAPI test client instance."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check returns 200 and system details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "api_key_configured" in data
    assert "default_model" in data
    assert data["supported_models_count"] > 0


def test_list_models_endpoint(client):
    """Test retrieving supported models catalog."""
    response = client.get("/api/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 5
    model_ids = [m["id"] for m in models]
    assert "gemini-3.5-flash" in model_ids


def test_analyze_endpoint(client):
    """Test text analysis endpoint."""
    sample_text = "Artificial intelligence is evolving at a rapid pace. " * 10
    response = client.post("/api/analyze", json={"text": sample_text})
    assert response.status_code == 200
    data = response.json()
    assert data["word_count"] > 50
    assert data["character_count"] > 0
    assert data["estimated_tokens"] > 0
    assert "reading_time_minutes" in data
    assert "is_long_article" in data


def test_extract_invalid_url(client):
    """Test extract endpoint with invalid URL returns 400."""
    response = client.post("/api/extract", json={"url": "not-a-valid-url"})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_upload_file_endpoint(client):
    """Test uploading a text file."""
    content = b"This is a test article content for the file upload endpoint. " * 5
    file_payload = {
        "file": ("test_article.txt", io.BytesIO(content), "text/plain")
    }
    response = client.post("/api/upload", files=file_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Article"
    assert "test article content" in data["text"].lower()
    assert data["word_count"] > 20


def test_upload_invalid_file_extension(client):
    """Test uploading an unsupported file type returns 400."""
    file_payload = {
        "file": ("test_executable.exe", io.BytesIO(b"fake binary"), "application/octet-stream")
    }
    response = client.post("/api/upload", files=file_payload)
    assert response.status_code == 400


@patch("src.api.app.create_gemini_client")
@patch("src.api.app.generate_summary")
def test_summarize_endpoint_success(mock_generate, mock_create_client, client):
    """Test successful article summarization via POST /api/summarize."""
    mock_create_client.return_value = MagicMock()
    mock_generate.return_value = {
        "text": "## Summary\n\nThis is a mocked high-quality summary.",
        "usage_metadata": MagicMock(prompt_token_count=100, candidates_token_count=50, total_token_count=150),
    }

    payload = {
        "text": "This is an article explaining how modern deep learning works. " * 8,
        "title_hint": "Modern Deep Learning",
        "model_id": "gemini-3.5-flash",
        "output_format": "markdown",
        "save_to_disk": False,
    }
    response = client.post("/api/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "mocked high-quality summary" in data["summary"]
    assert data["model_used"] == "gemini-3.5-flash"
    assert "metrics" in data


def test_history_endpoint(client):
    """Test retrieving outputs history."""
    response = client.get("/api/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_root_serves_frontend(client):
    """Test root URL serves Vite frontend index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AI Article Summarizer" in response.text

