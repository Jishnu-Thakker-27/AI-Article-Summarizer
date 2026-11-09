"""FastAPI web application for AI Article Summarizer.

Exposes REST and streaming SSE endpoints for article extraction, text analysis,
Gemini-powered summarization, and history management.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    HistoryItem,
    ModelInfo,
    StructuredSummaryData,
    SummarizeRequest,
    SummarizeResponse,
)
from src.article_extractor import extract_from_url
from src.article_processor import analyze_article, chunk_text, clean_and_normalize_text
from src.config import (
    BASE_DIR,
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    SUPPORTED_MODELS,
    get_api_key,
    setup_logging,
)
from src.exceptions import (
    APIKeyError,
    ArticleExtractionError,
    ContentBlockedError,
    ModelNotFoundError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    SummarizerException,
    ValidationError,
)
from src.file_handler import save_summary_markdown
from src.gemini_service import (
    create_gemini_client,
    generate_summary,
    summarize_long_article_map_reduce,
)
from src.performance import PerformanceTracker
from src.prompt_templates import (
    build_summarization_prompt,
    parse_structured_output,
)
from src.validators import validate_model_id, validate_url

logger = setup_logging()

app = FastAPI(
    title="AI Article Summarizer API",
    description="High-performance AI Article Summarization powered by Google Gemini and FastAPI",
    version="1.0.0",
)

# Enable CORS for frontend development (Vite runs on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SummarizerException)
async def handle_summarizer_exception(request: Request, exc: SummarizerException):
    """Map internal domain exceptions to appropriate HTTP status codes."""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    if isinstance(exc, (ValidationError, ArticleExtractionError)):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, APIKeyError):
        status_code = status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, ModelNotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, RateLimitError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, QuotaExceededError):
        status_code = status.HTTP_402_PAYMENT_REQUIRED
    elif isinstance(exc, ContentBlockedError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, NetworkError):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.message,
            "hint": getattr(exc, "hint", None),
            "type": exc.__class__.__name__,
        },
    )


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API server health, API key presence, and active default model."""
    api_key = get_api_key()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        api_key_configured=bool(api_key),
        default_model=DEFAULT_MODEL_ID,
        supported_models_count=len(SUPPORTED_MODELS),
    )


@app.get("/api/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models() -> List[ModelInfo]:
    """Return all catalogued Gemini models and their roles."""
    result: List[ModelInfo] = []
    for model_data in SUPPORTED_MODELS.values():
        result.append(
            ModelInfo(
                id=model_data["id"],
                name=model_data["name"],
                description=model_data["description"],
                preference=model_data["preference"],
                intended_use=model_data["intended_use"],
                max_output_tokens=model_data["max_output_tokens"],
                is_default=(model_data["id"] == DEFAULT_MODEL_ID),
            )
        )
    return result


@app.post("/api/extract", response_model=ExtractResponse, tags=["Article"])
async def extract_article_url(req: ExtractRequest) -> ExtractResponse:
    """Extract clean title and article body text from an HTTP/HTTPS URL."""
    cleaned_url = validate_url(req.url)
    extracted = extract_from_url(cleaned_url)
    analysis = analyze_article(extracted.text)

    return ExtractResponse(
        title=extracted.title,
        text=extracted.text,
        url=extracted.url,
        word_count=analysis["word_count"],
        character_count=analysis["character_count"],
        estimated_tokens=analysis["estimated_tokens"],
    )


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["Article"])
async def analyze_text(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze raw article text for word count, tokens, and chunk requirements."""
    analysis = analyze_article(req.text)
    words = analysis["word_count"]
    reading_time = round(words / 200.0, 1)
    chunk_count = len(chunk_text(analysis["cleaned_text"])) if analysis["requires_map_reduce"] else 1

    return AnalyzeResponse(
        word_count=words,
        character_count=analysis["character_count"],
        estimated_tokens=analysis["estimated_tokens"],
        reading_time_minutes=max(0.1, reading_time),
        is_long_article=analysis["requires_map_reduce"],
        chunk_count=chunk_count,
    )


@app.post("/api/upload", response_model=ExtractResponse, tags=["Article"])
async def upload_file(file: UploadFile = File(...)) -> ExtractResponse:
    """Upload a plain text or Markdown article file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename")

    valid_extensions = {".txt", ".md", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Only .txt and .md files are allowed.",
        )

    content_bytes = await file.read()
    try:
        raw_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = content_bytes.decode("latin-1", errors="replace")

    cleaned_text = clean_and_normalize_text(raw_text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    derived_title = Path(file.filename).stem.replace("_", " ").replace("-", " ").title()
    analysis = analyze_article(cleaned_text)

    return ExtractResponse(
        title=derived_title,
        text=cleaned_text,
        url=f"file://{file.filename}",
        word_count=analysis["word_count"],
        character_count=analysis["character_count"],
        estimated_tokens=analysis["estimated_tokens"],
    )


@app.post("/api/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize_article(req: SummarizeRequest) -> SummarizeResponse:
    """Generate a high-quality summary for an article via Google Gemini."""
    model_id = req.model_id or DEFAULT_MODEL_ID
    model_id = validate_model_id(model_id, SUPPORTED_MODELS)

    client = create_gemini_client()
    text = clean_and_normalize_text(req.text)
    analysis = analyze_article(text)

    use_map_reduce = req.force_map_reduce or analysis["requires_map_reduce"]
    tracker = PerformanceTracker(input_word_count=analysis["word_count"])

    structured_data: Optional[StructuredSummaryData] = None
    saved_path_str: Optional[str] = None
    output_fmt = req.output_format.lower()
    if output_fmt not in ("markdown", "json"):
        output_fmt = "markdown"

    if use_map_reduce:
        result = summarize_long_article_map_reduce(
            client=client,
            article_text=text,
            model_id=model_id,
            output_format=output_fmt,
            performance_tracker=tracker,
        )
        summary_text = result["summary"]
        method_used = "map_reduce"
    else:
        prompt = build_summarization_prompt(
            article_text=text,
            title_hint=req.title_hint,
            output_format=output_fmt,
        )
        tracker.record_first_chunk()
        result = generate_summary(
            client=client,
            prompt=prompt,
            model_id=model_id,
        )
        summary_text = result["text"]
        tracker.record_completion(generated_text=summary_text)
        tracker.record_usage_metadata(result.get("usage_metadata"))
        method_used = "standard"

    # Attempt structured JSON parsing
    if output_fmt == "json":
        parsed = parse_structured_output(summary_text)
        if parsed:
            structured_data = StructuredSummaryData(
                title=parsed.title,
                tldr=parsed.tldr,
                summary=parsed.summary,
                key_takeaways=parsed.key_takeaways,
                topics_and_keywords=parsed.topics_and_keywords,
                central_message=parsed.central_message,
            )

    metrics = tracker.metrics.to_dict()

    # Save to disk if requested
    if req.save_to_disk:
        try:
            saved_file = save_summary_markdown(
                summary_text=summary_text,
                article_url=req.url,
                model_used=model_id,
                metrics=metrics,
                title_hint=req.title_hint or (structured_data.title if structured_data else None),
            )
            saved_path_str = str(saved_file)
        except Exception as exc:
            logger.warning("Could not auto-save summary to disk: %s", exc)

    return SummarizeResponse(
        success=True,
        summary=summary_text,
        structured=structured_data,
        model_used=model_id,
        method=method_used,
        saved_file=saved_path_str,
        metrics=metrics,
    )


@app.post("/api/summarize/stream", tags=["Summarization"])
async def summarize_stream_endpoint(req: SummarizeRequest):
    """Stream Gemini summary generation in real-time via Server-Sent Events (SSE)."""
    model_id = req.model_id or DEFAULT_MODEL_ID
    model_id = validate_model_id(model_id, SUPPORTED_MODELS)
    client = create_gemini_client()

    text = clean_and_normalize_text(req.text)
    analysis = analyze_article(text)

    tracker = PerformanceTracker(input_word_count=analysis["word_count"])

    output_fmt = req.output_format.lower()
    if output_fmt not in ("markdown", "json"):
        output_fmt = "markdown"

    prompt = build_summarization_prompt(
        article_text=text,
        title_hint=req.title_hint,
        output_format=output_fmt,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        full_text_parts: List[str] = []
        try:
            yield f"data: {json.dumps({'event': 'start', 'model': model_id})}\n\n"

            # Stream from Gemini
            response_stream = client.models.generate_content_stream(
                model=model_id,
                contents=prompt,
            )

            for chunk in response_stream:
                chunk_text = chunk.text or ""
                if chunk_text:
                    tracker.record_first_chunk()
                    full_text_parts.append(chunk_text)
                    yield f"data: {json.dumps({'event': 'chunk', 'text': chunk_text})}\n\n"

            full_summary = "".join(full_text_parts)
            tracker.record_completion(generated_text=full_summary)
            metrics = tracker.metrics.to_dict()

            # Attempt JSON parse if requested
            structured_dict = None
            if output_fmt == "json":
                parsed = parse_structured_output(full_summary)
                if parsed:
                    structured_dict = parsed.model_dump()

            # Save to disk
            saved_file_str = None
            if req.save_to_disk:
                try:
                    sf = save_summary_markdown(
                        summary_text=full_summary,
                        article_url=req.url,
                        model_used=model_id,
                        metrics=metrics,
                        title_hint=req.title_hint,
                    )
                    saved_file_str = str(sf)
                except Exception as save_err:
                    logger.warning("Failed saving streamed summary: %s", save_err)

            yield f"data: {json.dumps({'event': 'done', 'summary': full_summary, 'structured': structured_dict, 'metrics': metrics, 'saved_file': saved_file_str})}\n\n"

        except Exception as exc:
            logger.error("Error during streaming: %s", exc)
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/history", response_model=List[HistoryItem], tags=["History"])
async def list_history() -> List[HistoryItem]:
    """Retrieve list of saved summary markdown files from outputs/ directory."""
    if not OUTPUT_DIR.exists():
        return []

    items: List[HistoryItem] = []
    files = sorted(OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    for f in files[:30]:  # Limit to 30 most recent files
        stat = f.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        # Try to extract the first header as title
        title = f.stem
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                first_line = fh.readline()
                if first_line.startswith("# "):
                    title = first_line.replace("# ", "").strip()
        except Exception:
            pass

        items.append(
            HistoryItem(
                filename=f.name,
                path=str(f),
                size_bytes=stat.st_size,
                modified_at=mod_time,
                title=title,
            )
        )

    return items


@app.get("/api/history/{filename}", tags=["History"])
async def get_history_file(filename: str):
    """Download or view a specific saved markdown summary file."""
    # Prevent directory traversal
    safe_filename = Path(filename).name
    file_path = OUTPUT_DIR / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        media_type="text/markdown",
        filename=safe_filename,
    )


# Mount built frontend files if available (Vite build output)
FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
