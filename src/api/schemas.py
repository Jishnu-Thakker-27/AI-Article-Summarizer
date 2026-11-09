"""Pydantic schemas for the FastAPI endpoints of AI Article Summarizer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and configuration status."""

    status: str = Field(default="healthy", json_schema_extra={"example": "healthy"})
    version: str = Field(default="1.0.0", json_schema_extra={"example": "1.0.0"})
    api_key_configured: bool = Field(default=False, json_schema_extra={"example": True})
    default_model: str = Field(default="gemini-3.5-flash", json_schema_extra={"example": "gemini-3.5-flash"})
    supported_models_count: int = Field(default=5, json_schema_extra={"example": 5})


class ModelInfo(BaseModel):
    """Metadata for a supported Gemini model."""

    id: str
    name: str
    description: str
    preference: str
    intended_use: str
    max_output_tokens: int
    is_default: bool = False


class ExtractRequest(BaseModel):
    """Request payload to extract article content from a web URL."""

    url: str = Field(..., min_length=4, description="HTTP/HTTPS URL of the article to extract")


class ExtractResponse(BaseModel):
    """Response containing extracted article text and metadata."""

    title: str
    text: str
    url: str
    word_count: int
    character_count: int
    estimated_tokens: int


class AnalyzeRequest(BaseModel):
    """Request payload to analyze raw text characteristics."""

    text: str = Field(..., min_length=1, description="Raw article text to analyze")


class AnalyzeResponse(BaseModel):
    """Text structural and length metrics."""

    word_count: int
    character_count: int
    estimated_tokens: int
    reading_time_minutes: float
    is_long_article: bool
    chunk_count: int


class SummarizeRequest(BaseModel):
    """Request payload for summarizing article text."""

    text: str = Field(..., min_length=10, description="Article body text to summarize")
    title_hint: Optional[str] = Field(None, description="Optional known title")
    url: Optional[str] = Field(None, description="Source article URL if applicable")
    model_id: Optional[str] = Field(None, description="Gemini model ID to use")
    output_format: str = Field("markdown", description="'markdown' or 'json'")
    force_map_reduce: bool = Field(False, description="Force Map-Reduce even for shorter text")
    save_to_disk: bool = Field(True, description="Save generated markdown to outputs/ directory")


class StructuredSummaryData(BaseModel):
    """Structured fields parsed from JSON output mode."""

    title: str
    tldr: str
    summary: str
    key_takeaways: List[str]
    topics_and_keywords: List[str]
    central_message: str


class SummarizeResponse(BaseModel):
    """Response containing generated summary and performance metrics."""

    success: bool = True
    summary: str
    structured: Optional[StructuredSummaryData] = None
    model_used: str
    method: str
    saved_file: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class HistoryItem(BaseModel):
    """Information about a previously saved summary."""

    filename: str
    path: str
    size_bytes: int
    modified_at: str
    title: Optional[str] = None
