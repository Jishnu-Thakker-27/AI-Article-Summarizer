"""Gemini API communication service using the official google-genai SDK.

Handles client initialization, model querying, streaming content generation,
latency/token profiling, map-reduce summarization for long documents, and secure error mapping.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from google import genai
from google.genai import errors, types

from src.article_processor import chunk_text
from src.config import (
    DEFAULT_MODEL_ID,
    GEMINI_HTTP_TIMEOUT_MS,
    SUPPORTED_MODELS,
    get_api_key,
    mask_secret,
)
from src.exceptions import (
    APIKeyError,
    ContentBlockedError,
    ModelNotFoundError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    SummarizerException,
)
from src.performance import PerformanceTracker
from src.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_chunk_summary_prompt,
    build_reduce_summary_prompt,
)
from src.validators import validate_api_key, validate_model_id

logger = logging.getLogger("ai_article_summarizer.gemini")


def create_gemini_client(
    api_key: Optional[str] = None,
    timeout_ms: int = GEMINI_HTTP_TIMEOUT_MS,
) -> genai.Client:
    """Create and configure an official Google GenAI client instance.

    Args:
        api_key: Optional explicit Gemini API key. Defaults to environment config.
        timeout_ms: Request timeout in milliseconds.

    Returns:
        Configured genai.Client instance.

    Raises:
        APIKeyError: If the API key is missing or invalid.
    """
    raw_key = api_key if api_key is not None else get_api_key()
    valid_key = validate_api_key(raw_key)

    logger.debug("Creating Gemini client with API key: %s", mask_secret(valid_key))

    try:
        client = genai.Client(
            api_key=valid_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        return client
    except Exception as exc:
        logger.error("Failed to initialize Gemini Client: %s", exc)
        raise map_api_exception(exc) from exc


def get_available_models(client: genai.Client) -> List[Dict[str, Any]]:
    """Retrieve available Gemini models from Google AI or fall back to known catalog.

    Args:
        client: Active Gemini client.

    Returns:
        List of dictionaries containing model metadata.
    """
    models_list: List[Dict[str, Any]] = []

    try:
        remote_models = client.models.list()
        # Filter for models containing 'gemini' and supporting text generation
        for m in remote_models:
            name = getattr(m, "name", "")
            clean_id = name.replace("models/", "")
            if "gemini" in clean_id and not any(
                skip in clean_id for skip in ["embedding", "robotics", "image", "tts"]
            ):
                models_list.append({
                    "id": clean_id,
                    "name": clean_id,
                    "description": getattr(m, "description", "Google Gemini model"),
                    "display_name": getattr(m, "display_name", clean_id),
                })
    except Exception as exc:
        logger.warning("Could not list remote models from API (%s). Using local catalog.", exc)

    # If remote listing produced models, return them; otherwise use local configuration
    if models_list:
        return models_list

    return [
        {
            "id": m["id"],
            "name": m["name"],
            "description": m["description"],
            "preference": m["preference"],
            "intended_use": m["intended_use"],
        }
        for m in SUPPORTED_MODELS.values()
    ]


def map_api_exception(exc: Exception) -> SummarizerException:
    """Map Google GenAI SDK exceptions to application-specific exceptions.

    Guarantees that sensitive credentials are never exposed in log outputs or UI.

    Args:
        exc: Exception caught from SDK operation.

    Returns:
        Appropriate instance of SummarizerException subclass.
    """
    # Check for HTTP status code on APIError or ClientError
    status_code = getattr(exc, "code", None)
    error_message = str(exc)

    if isinstance(exc, errors.ClientError) or status_code is not None:
        if status_code in (401, 403):
            return APIKeyError(
                message="Authentication failed: Gemini API key is invalid or lacks necessary permissions.",
                hint="Verify that your GEMINI_API_KEY in .env is correct and has access to Gemini models.",
            )
        elif status_code == 404:
            return ModelNotFoundError(
                model_id="Selected Model",
                hint="This model is not available or has been deprecated. Try 'gemini-3.5-flash'.",
            )
        elif status_code == 429:
            if "quota" in error_message.lower():
                return QuotaExceededError(
                    message="Gemini project quota limit reached.",
                    hint="Check usage quotas at https://aistudio.google.com/app/plan_information",
                )
            return RateLimitError(
                message="Gemini API rate limit exceeded.",
                hint="Please wait a moment before sending another request.",
            )

    if isinstance(exc, errors.ServerError) or status_code in (500, 502, 503, 504):
        return SummarizerException(
            message=f"Gemini service is temporarily unavailable (HTTP {status_code or 503}).",
            hint="Google's servers are experiencing temporary load. Please try again shortly or switch models.",
        )

    if isinstance(exc, TimeoutError) or "timeout" in error_message.lower():
        return NetworkError(
            message="Request to Gemini API timed out.",
            hint="Check your network connection and try again.",
        )

    return SummarizerException(
        message=f"Gemini API request failed: {error_message}",
        hint="Check input parameters and try again.",
    )


def generate_summary(
    client: genai.Client,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    system_instruction: str = SYSTEM_INSTRUCTION,
    max_output_tokens: int = 1500,
) -> Dict[str, Any]:
    """Generate a summary in a single synchronous call.

    Args:
        client: Initialized Gemini client.
        prompt: User prompt content.
        model_id: Target Gemini model ID.
        system_instruction: System prompt enforcing role and safety rules.
        max_output_tokens: Maximum tokens for response.

    Returns:
        Dictionary containing generated text and usage metadata.
    """
    clean_model = validate_model_id(model_id, SUPPORTED_MODELS)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=0.2,  # Low temperature for factual fidelity
    )

    try:
        response = client.models.generate_content(
            model=clean_model,
            contents=prompt,
            config=config,
        )

        # Inspect safety finish reason
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason and "SAFETY" in str(finish_reason).upper():
                raise ContentBlockedError()

        text = response.text or ""
        if not text.strip():
            raise SummarizerException(
                "Gemini returned an empty response.",
                hint="The article content might be too short or triggered an unlisted safety block.",
            )

        return {
            "text": text,
            "usage_metadata": getattr(response, "usage_metadata", None),
        }
    except Exception as exc:
        raise map_api_exception(exc) from exc


def generate_summary_stream(
    client: genai.Client,
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    system_instruction: str = SYSTEM_INSTRUCTION,
    max_output_tokens: int = 1500,
    on_chunk: Optional[Callable[[str], None]] = None,
    performance_tracker: Optional[PerformanceTracker] = None,
) -> Dict[str, Any]:
    """Generate a summary with live streaming of response chunks.

    Measures Time to First Token (TTFT) and token usage progressively.

    Args:
        client: Initialized Gemini client.
        prompt: User prompt.
        model_id: Target model ID.
        system_instruction: System instruction string.
        max_output_tokens: Token generation budget.
        on_chunk: Callback invoked each time a new text chunk arrives.
        performance_tracker: Performance tracker instance for recording latencies.

    Returns:
        Dictionary with full generated text and performance metrics.
    """
    clean_model = validate_model_id(model_id, SUPPORTED_MODELS)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=0.2,
    )

    generated_parts: List[str] = []
    last_chunk_metadata = None

    try:
        response_stream = client.models.generate_content_stream(
            model=clean_model,
            contents=prompt,
            config=config,
        )

        for chunk in response_stream:
            # Check for safety filter blocks on chunk
            if chunk.candidates:
                candidate = chunk.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason and "SAFETY" in str(finish_reason).upper():
                    raise ContentBlockedError()

            if chunk.text:
                if performance_tracker:
                    performance_tracker.record_first_chunk()
                generated_parts.append(chunk.text)
                if on_chunk:
                    on_chunk(chunk.text)

            # Capture usage metadata from the final chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_chunk_metadata = chunk.usage_metadata

        full_text = "".join(generated_parts).strip()
        if not full_text:
            raise SummarizerException(
                "Gemini returned an empty response stream.",
                hint="Check input article contents or try a different Gemini model.",
            )

        if performance_tracker:
            performance_tracker.record_usage_metadata(last_chunk_metadata)
            performance_tracker.record_completion(full_text)

        return {
            "text": full_text,
            "usage_metadata": last_chunk_metadata,
            "model": clean_model,
        }
    except Exception as exc:
        raise map_api_exception(exc) from exc


def summarize_long_article_map_reduce(
    client: genai.Client,
    article_text: str,
    model_id: str = DEFAULT_MODEL_ID,
    title_hint: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    on_stream_chunk: Optional[Callable[[str], None]] = None,
    performance_tracker: Optional[PerformanceTracker] = None,
) -> Dict[str, Any]:
    """Execute Map-Reduce summarization strategy for long articles.

    Divides the article into manageable chunks, summarizes each chunk independently
    (Map phase), aggregates the partial summaries, and synthesizes a master summary (Reduce phase).

    Args:
        client: Initialized Gemini client.
        article_text: Full article text exceeding standard single-pass limits.
        model_id: Target Gemini model ID.
        title_hint: Optional extracted title.
        on_progress: Progress update callback for terminal status.
        on_stream_chunk: Chunk callback for the final synthesis phase.
        performance_tracker: Performance tracker instance.

    Returns:
        Dictionary containing the master summary and metadata.
    """
    clean_model = validate_model_id(model_id, SUPPORTED_MODELS)
    chunks = chunk_text(article_text)
    total_chunks = len(chunks)

    if on_progress:
        on_progress(f"Long article detected. Processing in {total_chunks} sections using Map-Reduce...")

    # MAP PHASE: Summarize each section
    intermediate_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if on_progress:
            on_progress(f"Analyzing section {idx} of {total_chunks}...")

        chunk_prompt = build_chunk_summary_prompt(
            chunk_text=chunk,
            chunk_index=idx,
            total_chunks=total_chunks,
        )

        res = generate_summary(
            client=client,
            prompt=chunk_prompt,
            model_id=clean_model,
            max_output_tokens=600,
        )
        intermediate_summaries.append(f"Section {idx} Summary:\n{res['text']}")

    # REDUCE PHASE: Master synthesis streamed live
    if on_progress:
        on_progress("Synthesizing master summary from all sections...")

    combined_notes = "\n\n".join(intermediate_summaries)
    reduce_prompt = build_reduce_summary_prompt(combined_notes, title_hint=title_hint)

    master_result = generate_summary_stream(
        client=client,
        prompt=reduce_prompt,
        model_id=clean_model,
        max_output_tokens=2000,
        on_chunk=on_stream_chunk,
        performance_tracker=performance_tracker,
    )

    return master_result
