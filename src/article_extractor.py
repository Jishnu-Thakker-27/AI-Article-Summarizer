"""Webpage article content extraction using requests and BeautifulSoup4.

Extracts titles and main body copy from HTTP/HTTPS URLs with semantic element priority,
polite robots.txt checks, ad/nav removal, and clear fallback guidance when scraping is prevented.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.config import DEFAULT_USER_AGENT, MIN_ARTICLE_WORDS, REQUEST_TIMEOUT_SECONDS
from src.exceptions import ArticleExtractionError
from src.validators import validate_url

logger = logging.getLogger("ai_article_summarizer.extractor")


@dataclass
class ExtractedArticle:
    """Container for successfully extracted webpage article data."""

    title: str
    text: str
    url: str
    word_count: int


def is_allowed_by_robots(url: str, user_agent: str = DEFAULT_USER_AGENT, timeout: int = 5) -> bool:
    """Check if the given URL is allowed by the website's robots.txt policy.

    Args:
        url: The webpage URL.
        user_agent: User-Agent header identifier.
        timeout: Timeout for fetching robots.txt in seconds.

    Returns:
        True if scraping is permitted or if robots.txt is absent/unreachable;
        False if explicitly disallowed.
    """
    try:
        parsed = urlparse(url)
        robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
        parser = RobotFileParser()
        parser.set_url(robots_url)
        
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=timeout)
        if resp.status_code == 200:
            parser.parse(resp.text.splitlines())
            allowed = parser.can_fetch(user_agent, url)
            if not allowed:
                logger.warning("Scraping disallowed by robots.txt: %s", url)
            return allowed
    except Exception as exc:
        logger.debug("robots.txt check skipped due to error: %s", exc)
    return True


def extract_title(soup: BeautifulSoup) -> str:
    """Extract the most representative article title from HTML.

    Inspects OpenGraph meta tags, Twitter card tags, <title>, and <h1> elements.

    Args:
        soup: BeautifulSoup parse tree.

    Returns:
        Extracted title string or 'Untitled Article'.
    """
    # 1. OpenGraph title
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # 2. Twitter card title
    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    if tw_title and tw_title.get("content"):
        return tw_title["content"].strip()

    # 3. HTML <title> tag
    if soup.title and soup.title.string:
        raw_title = soup.title.string.strip()
        # Clean common branding suffixes: 'Article Name | Publication' or 'Article Name - Brand'
        cleaned = re.split(r"\s+[|\-—–]\s+", raw_title)[0]
        if cleaned:
            return cleaned.strip()

    # 4. First <h1> element
    h1 = soup.find("h1")
    if h1 and h1.get_text():
        return h1.get_text().strip()

    return "Untitled Article"


def extract_article_content(soup: BeautifulSoup) -> str:
    """Extract clean body copy from semantic HTML elements with fallback.

    Removes script, style, nav, footer, sidebar, and advertisement containers,
    prioritizing <article> and <main> tags before falling back to general paragraphs.

    Args:
        soup: BeautifulSoup parse tree.

    Returns:
        Extracted article body text separated by double newlines.
    """
    # Clone soup to avoid mutating original
    working_soup = BeautifulSoup(str(soup), "html.parser")

    # Decompose unwanted elements
    unwanted_tags = [
        "script",
        "style",
        "noscript",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "svg",
        "button",
        "iframe",
        "figure",
        "figcaption",
        "dialog",
        "time",
    ]
    for tag in working_soup.find_all(unwanted_tags):
        tag.decompose()

    # Decompose elements with ad, sidebar, cookie, social, or comment classes/ids
    ad_pattern = re.compile(
        r"(advertisement|ad-container|ad-wrapper|sidebar|cookie|banner|newsletter|social-share|comments)",
        re.I,
    )
    for el in working_soup.find_all(attrs={"class": ad_pattern}):
        el.decompose()
    for el in working_soup.find_all(attrs={"id": ad_pattern}):
        el.decompose()

    # Priority 1: Semantic <article> element
    article_tag = working_soup.find("article")
    if article_tag:
        paras = [p.get_text().strip() for p in article_tag.find_all("p") if len(p.get_text().strip()) > 25]
        if len(" ".join(paras).split()) >= MIN_ARTICLE_WORDS:
            return "\n\n".join(paras)

    # Priority 2: Semantic <main> element
    main_tag = working_soup.find("main") or working_soup.find(attrs={"role": "main"})
    if main_tag:
        paras = [p.get_text().strip() for p in main_tag.find_all("p") if len(p.get_text().strip()) > 25]
        if len(" ".join(paras).split()) >= MIN_ARTICLE_WORDS:
            return "\n\n".join(paras)

    # Priority 3: Common content container classes
    container_pattern = re.compile(
        r"(entry-content|post-content|article-body|story-body|content-body)",
        re.I,
    )
    content_div = working_soup.find(attrs={"class": container_pattern})
    if content_div:
        paras = [p.get_text().strip() for p in content_div.find_all("p") if len(p.get_text().strip()) > 25]
        if len(" ".join(paras).split()) >= MIN_ARTICLE_WORDS:
            return "\n\n".join(paras)

    # Priority 4: Fallback to all substantial <p> tags in the document body
    all_paras = [p.get_text().strip() for p in working_soup.find_all("p") if len(p.get_text().strip()) > 35]
    return "\n\n".join(all_paras)


def extract_from_url(
    url: str,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    check_robots: bool = True,
) -> ExtractedArticle:
    """Fetch and extract article content and metadata from a web URL.

    Args:
        url: Target web page URL.
        timeout: Network request timeout in seconds.
        check_robots: Whether to respect robots.txt rules.

    Returns:
        ExtractedArticle dataclass containing title, clean text, url, and word count.

    Raises:
        ArticleExtractionError: If network fails, URL is blocked, or content is insufficient.
    """
    clean_url = validate_url(url)

    if check_robots and not is_allowed_by_robots(clean_url):
        raise ArticleExtractionError(
            f"Access to '{clean_url}' is restricted by the site's robots.txt policy.",
            hint="Please copy and paste the article text directly into the summarizer.",
        )

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(clean_url, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise ArticleExtractionError(
            f"Request timed out after {timeout} seconds connecting to {clean_url}.",
            hint="The target website may be slow or unresponsive. Check the URL or try pasting the text directly.",
        ) from exc
    except requests.exceptions.SSLError as exc:
        raise ArticleExtractionError(
            f"SSL certificate validation failed for {clean_url}: {exc}",
            hint="The site has an invalid HTTPS configuration. Try copying and pasting the text.",
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ArticleExtractionError(
            f"Failed to connect to '{clean_url}': {exc}",
            hint="Check your internet connection and verify that the URL is accessible in your browser.",
        ) from exc

    if response.status_code == 403 or response.status_code == 401:
        raise ArticleExtractionError(
            f"Access denied (HTTP {response.status_code}) by {clean_url}.",
            hint="This page may be paywalled, require login credentials, or block automated bots. Please paste the article text.",
        )
    elif response.status_code == 404:
        raise ArticleExtractionError(
            f"Article page not found (HTTP 404): {clean_url}.",
            hint="Check the URL for typos and confirm the article exists.",
        )
    elif response.status_code >= 400:
        raise ArticleExtractionError(
            f"Web server responded with HTTP {response.status_code} for {clean_url}.",
            hint="Please verify the URL or copy and paste the text directly.",
        )

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise ArticleExtractionError(
            f"Unsupported content type '{content_type}'. Expected HTML webpage.",
            hint="This tool extracts web articles (HTML). For PDFs or binary files, extract text first.",
        )

    soup = BeautifulSoup(response.text, "html.parser")
    title = extract_title(soup)
    text = extract_article_content(soup)
    word_count = len(text.split())

    if word_count < MIN_ARTICLE_WORDS:
        raise ArticleExtractionError(
            f"Insufficient article content extracted ({word_count} words found).",
            hint=(
                "The article may be rendered client-side using JavaScript (e.g. React/Vue), "
                "behind an anti-bot CAPTCHA, or behind a paywall. "
                "Please copy the article text from your browser and select option 1 to paste it."
            ),
        )

    return ExtractedArticle(
        title=title,
        text=text,
        url=clean_url,
        word_count=word_count,
    )
