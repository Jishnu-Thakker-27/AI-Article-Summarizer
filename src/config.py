"""Centralized configuration management for AI Article Summarizer.

Provides application paths, environment variable loading, model catalogs,
text processing thresholds, network settings, and secure logging helpers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Base paths
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure .env is loaded
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# Network and API timeouts
REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
GEMINI_HTTP_TIMEOUT_MS: int = int(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "60000"))
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 AI-Article-Summarizer/1.0"
)

# Text thresholds for analysis and chunking
MIN_ARTICLE_WORDS: int = 30
UNUSUALLY_LARGE_WORD_THRESHOLD: int = 20_000
MAP_REDUCE_THRESHOLD_WORDS: int = 3500
CHUNK_SIZE_WORDS: int = 2500
CHUNK_OVERLAP_WORDS: int = 200

# Centralized catalog of supported Gemini models
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "1": {
        "id": "gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "description": "Fast, balanced multimodal model with high quality",
        "intended_use": "Default summarization, quick and accurate",
        "preference": "Speed & Quality Balanced",
        "max_output_tokens": 1500,
    },
    "2": {
        "id": "gemini-3.5-flash-lite",
        "name": "Gemini 3.5 Flash-Lite",
        "description": "Extremely fast, cost-efficient lightweight model",
        "intended_use": "Short to medium articles, high throughput",
        "preference": "Maximum Speed",
        "max_output_tokens": 1200,
    },
    "3": {
        "id": "gemini-3.8-flash",
        "name": "Gemini 3.8 Flash",
        "description": "Premier Flash tier with superior analytical depth and reasoning",
        "intended_use": "Complex, dense, scientific or technical articles",
        "preference": "Highest Quality",
        "max_output_tokens": 2000,
    },
    "4": {
        "id": "gemini-3.6-flash",
        "name": "Gemini 3.6 Flash",
        "description": "Reliable, well-rounded high speed flash model",
        "intended_use": "Standard articles, news, and essays",
        "preference": "Balanced",
        "max_output_tokens": 1500,
    },
    "5": {
        "id": "gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash-Lite",
        "description": "Lightweight model for low-latency summarization",
        "intended_use": "Basic summaries, lightweight environments",
        "preference": "Ultra Lightweight",
        "max_output_tokens": 1000,
    },
}

# Configurable default model ID
DEFAULT_MODEL_ID: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def get_api_key() -> str | None:
    """Retrieve the Gemini API key from the environment.

    Checks GEMINI_API_KEY first, followed by API_KEY for backward compatibility.
    Never logs the raw key value.

    Returns:
        The API key string if found, otherwise None.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if key:
        key = key.strip()
    return key or None


def mask_secret(secret: str | None, visible_chars: int = 4) -> str:
    """Safely mask a secret string for logging or UI display.

    Args:
        secret: The sensitive string (e.g., API key).
        visible_chars: Number of trailing characters to leave unmasked.

    Returns:
        Masked string (e.g. '****...abcd') or '<NOT SET>'.
    """
    if not secret:
        return "<NOT SET>"
    if len(secret) <= visible_chars:
        return "*" * len(secret)
    masked_length = max(4, len(secret) - visible_chars)
    return f"{'*' * masked_length}{secret[-visible_chars:]}"


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Configure application logging with a clean, structured format.

    Args:
        level: Optional logging level. Defaults to LOG_LEVEL env var or logging.WARNING.

    Returns:
        Configured Logger instance for the application.
    """
    if level is None:
        level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
        level = getattr(logging, level_str, logging.WARNING)

    logger = logging.getLogger("ai_article_summarizer")
    if not logger.handlers:
        logger.setLevel(level)
        import sys
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
