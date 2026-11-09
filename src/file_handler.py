"""File management and Markdown report generation for AI Article Summarizer.

Handles reading input text files with encoding fallbacks and writing formatted
Markdown summary artifacts to the outputs/ directory with timestamped collision avoidance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import OUTPUT_DIR
from src.exceptions import FileHandlingError
from src.validators import validate_file_path

logger = logging.getLogger("ai_article_summarizer.file_handler")


def read_text_file(file_path: Path | str) -> str:
    """Read article text from a local file with robust encoding detection.

    Args:
        file_path: Path to the target text file.

    Returns:
        Stripped string contents of the file.

    Raises:
        FileHandlingError: If the file cannot be read.
    """
    path_obj = validate_file_path(file_path)

    # Try UTF-8 first, fallback to common alternatives
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            content = path_obj.read_text(encoding=enc).strip()
            if content:
                logger.debug("Successfully read '%s' using %s encoding", path_obj.name, enc)
                return content
        except (UnicodeDecodeError, OSError) as exc:
            logger.debug("Failed reading '%s' with %s: %s", path_obj.name, enc, exc)
            continue

    raise FileHandlingError(
        f"Unable to decode text file '{path_obj.name}'.",
        hint="Please ensure the file is saved in a standard text format (UTF-8).",
    )


def generate_unique_filename(
    base_name: str = "summary",
    extension: str = ".md",
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate a collision-resistant timestamped file path.

    Format: {base_name}_YYYYMMDD_HHMMSS{_counter}.{extension}

    Args:
        base_name: Prefix string for the filename.
        extension: File extension including dot.
        output_dir: Directory where the file will be created.

    Returns:
        A unique, non-existing Path object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{base_name}_{timestamp}{extension}"

    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{timestamp}_{counter}{extension}"
        counter += 1

    return candidate


def save_summary_markdown(
    summary_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Path:
    """Save the generated summary and metadata as a formatted Markdown file.

    Complies with all portfolio metadata requirements:
    - Article Title
    - Source URL (if available)
    - Date and time
    - Selected Gemini model
    - Original word count
    - Generation duration
    - TL;DR
    - Summary
    - Key Takeaways (exactly three bullet points)
    - Important Topics & Keywords
    - Central Message
    - Generator attribution

    Args:
        summary_data: Dictionary containing parsed summary fields and performance metrics.
        output_dir: Target directory, defaults to config.OUTPUT_DIR.

    Returns:
        Path of the newly saved Markdown file.

    Raises:
        FileHandlingError: If the file cannot be written to disk.
    """
    target_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    target_path = generate_unique_filename(base_name="summary", extension=".md", output_dir=target_dir)

    title = summary_data.get("title", "Article Summary")
    source_url = summary_data.get("url")
    model_name = summary_data.get("model", "Gemini")
    input_words = summary_data.get("word_count", 0)
    duration = summary_data.get("total_duration_sec", 0.0)
    tldr = summary_data.get("tldr", "").strip()
    summary_body = summary_data.get("summary", "").strip()
    takeaways = summary_data.get("key_takeaways", [])
    topics = summary_data.get("topics_and_keywords", [])
    central_message = summary_data.get("central_message", "").strip()

    # Format takeaways
    if isinstance(takeaways, list):
        takeaways_formatted = "\n".join(f"- {item.strip()}" for item in takeaways[:3])
    else:
        takeaways_formatted = f"- {takeaways}"

    # Format topics
    if isinstance(topics, list):
        topics_formatted = ", ".join(topics)
    else:
        topics_formatted = str(topics)

    # Format metadata header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_line = f"- **Source URL:** [{source_url}]({source_url})" if source_url else "- **Source:** Pasted text / Local file"

    markdown_content = f"""# {title}

## Metadata
{source_line}
- **Date & Time:** {now_str}
- **Gemini Model:** `{model_name}`
- **Original Word Count:** {input_words} words
- **Generation Duration:** {duration:.2f} seconds

---

## TL;DR
{tldr or "No TL;DR provided."}

## Summary
{summary_body or "No summary content."}

## Key Takeaways
{takeaways_formatted or "- Key takeaway unavailable"}

## Important Topics & Keywords
{topics_formatted or "Not specified"}

## Central Message
{central_message or "Not specified"}

---
*Generated by AI Article Summarizer powered by Google Gemini.*
"""

    try:
        target_path.write_text(markdown_content.strip() + "\n", encoding="utf-8")
        logger.info("Saved summary report to: %s", target_path)
        return target_path
    except OSError as exc:
        raise FileHandlingError(
            f"Failed to write summary to '{target_path}': {exc}",
            hint="Check folder permissions and disk space.",
        ) from exc


def list_saved_summaries(output_dir: Optional[Path] = None) -> List[Path]:
    """Retrieve all markdown summary files in the output directory ordered by date.

    Args:
        output_dir: Directory to inspect.

    Returns:
        List of Path objects sorted newest first.
    """
    target = Path(output_dir) if output_dir else OUTPUT_DIR
    if not target.exists():
        return []
    files = list(target.glob("summary_*.md"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files
