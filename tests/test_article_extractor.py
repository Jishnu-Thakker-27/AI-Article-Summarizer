"""Unit tests for src.article_extractor module."""

from unittest.mock import MagicMock, patch
import pytest
from bs4 import BeautifulSoup

from src.article_extractor import (
    extract_article_content,
    extract_from_url,
    extract_title,
)
from src.exceptions import ArticleExtractionError


def test_extract_title_og_tag() -> None:
    """Test that OpenGraph title takes precedence."""
    html = """<html><head>
    <meta property="og:title" content="OpenGraph Article Title" />
    <title>Ignored HTML Title</title>
    </head><body><h1>H1 Title</h1></body></html>"""
    soup = BeautifulSoup(html, "html.parser")
    assert extract_title(soup) == "OpenGraph Article Title"


def test_extract_title_html_title_cleaned() -> None:
    """Test cleaning publication suffix from HTML title."""
    html = "<html><head><title>SpaceX Launches Starship | TechNews</title></head></html>"
    soup = BeautifulSoup(html, "html.parser")
    assert extract_title(soup) == "SpaceX Launches Starship"


def test_extract_article_content_semantic_article() -> None:
    """Test extracting body content from semantic <article> tags, stripping nav/scripts."""
    html = """
    <html>
      <body>
        <nav><a href="/">Home</a></nav>
        <div class="ad-container">Buy stuff now</div>
        <article>
          <script>console.log("bad");</script>
          <p>Artificial intelligence is rapidly transforming software engineering workflows across the globe today.</p>
          <p>Developers are utilizing advanced language models to automate routine code generation, refactoring, and testing.</p>
        </article>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    content = extract_article_content(soup)

    assert "Buy stuff" not in content
    assert "Home" not in content
    assert "Copyright" not in content
    assert "Artificial intelligence is rapidly transforming" in content
    assert "Developers are utilizing advanced language models" in content


@patch("src.article_extractor.requests.get")
@patch("src.article_extractor.is_allowed_by_robots", return_value=True)
def test_extract_from_url_success(mock_robots: MagicMock, mock_get: MagicMock) -> None:
    """Test end-to-end URL extraction with mocked HTTP response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_resp.text = """
    <html>
      <head><title>Deep Learning Breakthroughs</title></head>
      <body>
        <article>
          <p>Deep learning models continue to scale effectively across massive multimodal datasets in modern research labs and university computing clusters worldwide.</p>
          <p>Recent architectural innovations have substantially reduced computational requirements while boosting reasoning fidelity across complex contextual benchmarks.</p>
        </article>
      </body>
    </html>
    """
    mock_get.return_value = mock_resp

    result = extract_from_url("https://example.com/ai-news")
    assert result.title == "Deep Learning Breakthroughs"
    assert "Deep learning models continue to scale" in result.text
    assert result.word_count >= 30


@patch("src.article_extractor.requests.get")
@patch("src.article_extractor.is_allowed_by_robots", return_value=True)
def test_extract_from_url_403_forbidden(mock_robots: MagicMock, mock_get: MagicMock) -> None:
    """Test that HTTP 403 raises descriptive ArticleExtractionError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_get.return_value = mock_resp

    with pytest.raises(ArticleExtractionError) as exc:
        extract_from_url("https://paywalled-site.com/article")
    assert "access denied" in str(exc.value).lower()
