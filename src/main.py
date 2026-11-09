"""Command-line interface and main application controller for AI Article Summarizer.

Provides an interactive menu driven terminal interface as well as direct non-interactive
CLI execution using argparse.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure standard output uses UTF-8 on Windows terminals
if sys.platform == "win32":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.article_extractor import extract_from_url
from src.article_processor import analyze_article
from src.config import (
    DEFAULT_MODEL_ID,
    OUTPUT_DIR,
    SUPPORTED_MODELS,
    setup_logging,
)
from src.exceptions import (
    SummarizerException,
)
from src.file_handler import read_text_file, save_summary_markdown
from src.gemini_service import (
    create_gemini_client,
    generate_summary_stream,
    summarize_long_article_map_reduce,
)
from src.performance import PerformanceTracker
from src.prompt_templates import (
    build_summarization_prompt,
    parse_structured_output,
)

logger = setup_logging()


def print_banner() -> None:
    """Display the application startup banner."""
    print("\n" + "=" * 50)
    print("           AI ARTICLE SUMMARIZER")
    print("         Powered by Google Gemini")
    print("=" * 50 + "\n")


def display_app_info() -> None:
    """Display comprehensive application info and usage guide."""
    print("\n" + "-" * 50)
    print("APPLICATION INFORMATION")
    print("-" * 50)
    print("AI Article Summarizer is a modular, production-ready AI engineering tool")
    print("built with Python and Google's official GenAI SDK (google-genai).")
    print("\nKey Features:")
    print(" - Input flexibility: Paste text, extract from URL, or load from file")
    print(" - Structured summaries: TL;DR, 3 Takeaways, Topics, and Central Message")
    print(" - Real-time streaming with live terminal chunk output")
    print(" - Map-Reduce chunking for long articles exceeding context bounds")
    print(" - Precision latency and token usage performance tracking")
    print(" - Timestamped, collision-proof Markdown export in outputs/")
    print(" - Defense against prompt injections in untrusted source text")
    print("-" * 50 + "\n")


def display_models_catalog(client: Any = None) -> None:
    """Display available Gemini models with metadata."""
    print("\n" + "=" * 65)
    print("               AVAILABLE GEMINI MODELS")
    print("=" * 65)

    for key, model in SUPPORTED_MODELS.items():
        is_default = " (DEFAULT)" if model["id"] == DEFAULT_MODEL_ID else ""
        print(f"\n[{key}] {model['name']}{is_default}")
        print(f"    ID        : {model['id']}")
        print(f"    Role      : {model['intended_use']}")
        print(f"    Preference: {model['preference']}")
        print(f"    Info      : {model['description']}")

    print("\n" + "=" * 65 + "\n")


def prompt_model_selection(default_model: str = DEFAULT_MODEL_ID) -> str:
    """Prompt user to choose a Gemini model or accept default.

    Args:
        default_model: Default model ID.

    Returns:
        Selected model ID.
    """
    print(f"\nCurrent model: {default_model}")
    change = input("Press [Enter] to keep this model, or type 'c' to choose another: ").strip().lower()
    if change != "c":
        return default_model

    print("\nSelect a Gemini Model:")
    for key, model in SUPPORTED_MODELS.items():
        print(f"  [{key}] {model['name']} - {model['preference']}")

    while True:
        choice = input(f"\nSelect model (1-{len(SUPPORTED_MODELS)}) [1]: ").strip() or "1"
        if choice in SUPPORTED_MODELS:
            selected = SUPPORTED_MODELS[choice]["id"]
            print(f"Selected: {SUPPORTED_MODELS[choice]['name']} ({selected})")
            return selected
        print("Invalid selection. Please choose a valid number.")


def get_pasted_article() -> str:
    """Prompt user to paste multi-line article text from the terminal.

    Returns:
        Pasted article string.
    """
    print("\nPaste your article below.")
    print("Type 'END' on a new line and press Enter when finished:\n")

    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    article = "\n".join(lines).strip()
    return article


def run_summarization_pipeline(
    raw_text: str,
    model_id: str,
    source_url: Optional[str] = None,
    title_hint: Optional[str] = None,
    save_output: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end summarization pipeline with live streaming.

    Args:
        raw_text: Raw article text.
        model_id: Gemini model ID.
        source_url: Optional source URL if extracted from web.
        title_hint: Optional extracted title.
        save_output: Whether to persist markdown report to disk.
        output_dir: Target output directory.

    Returns:
        Dictionary containing parsed summary and performance data.
    """
    # 1. Structural Analysis & Normalization
    analysis = analyze_article(raw_text)
    cleaned_text = analysis["cleaned_text"]
    word_count = analysis["word_count"]
    estimated_tokens = analysis["estimated_tokens"]

    print("\n---------------- ARTICLE DETAILS ----------------")
    if title_hint:
        print(f"Title            : {title_hint}")
    if source_url:
        print(f"Source URL       : {source_url}")
    print(f"Word Count       : {word_count} words")
    print(f"Estimated Tokens : ~{estimated_tokens} tokens")
    print(f"Selected Model   : {model_id}")
    print("-------------------------------------------------")

    # 2. Initialize Gemini Client
    client = create_gemini_client()

    # 3. Setup Performance Tracker
    tracker = PerformanceTracker(input_word_count=word_count)

    print("\nGenerating summary...\n")
    print("=" * 60)

    # Callback to stream text directly to stdout
    def stream_callback(chunk_text: str) -> None:
        print(chunk_text, end="", flush=True)

    # 4. Long Document (Map-Reduce) vs Standard Single-Pass
    if analysis["requires_map_reduce"]:
        def progress_callback(msg: str) -> None:
            print(f"\n[MAP-REDUCE] {msg}")

        result_raw = summarize_long_article_map_reduce(
            client=client,
            article_text=cleaned_text,
            model_id=model_id,
            title_hint=title_hint,
            on_progress=progress_callback,
            on_stream_chunk=stream_callback,
            performance_tracker=tracker,
        )
    else:
        prompt = build_summarization_prompt(
            article_text=cleaned_text,
            title_hint=title_hint,
            output_format="markdown",
        )
        result_raw = generate_summary_stream(
            client=client,
            prompt=prompt,
            model_id=model_id,
            on_chunk=stream_callback,
            performance_tracker=tracker,
        )

    print("\n" + "=" * 60 + "\n")

    # 5. Parse and Structure the Output
    parsed_summary = parse_structured_output(result_raw["text"])
    if title_hint and (parsed_summary["title"] == "Article Summary" or not parsed_summary["title"]):
        parsed_summary["title"] = title_hint

    # 6. Performance Summary
    print(tracker.format_terminal_summary())

    # 7. Save to Markdown
    if save_output:
        combined_payload = {
            **parsed_summary,
            "url": source_url,
            "model": model_id,
            "word_count": word_count,
            "total_duration_sec": tracker.metrics.total_duration,
        }
        saved_file = save_summary_markdown(combined_payload, output_dir=output_dir)
        print(f"\n[SUCCESS] Summary report saved to: {saved_file}\n")

    return parsed_summary


def handle_summarize_pasted(selected_model: str) -> None:
    """Workflow for summarizing pasted text."""
    try:
        text = get_pasted_article()
        if not text:
            print("\n[WARNING] No text provided. Returning to main menu.")
            return
        model = prompt_model_selection(selected_model)
        run_summarization_pipeline(raw_text=text, model_id=model)
    except SummarizerException as exc:
        print(f"\n[ERROR] {exc}\n")


def handle_summarize_url(selected_model: str) -> None:
    """Workflow for extracting and summarizing a web article."""
    url = input("\nEnter article URL (e.g. https://example.com/post): ").strip()
    if not url:
        print("\n[WARNING] URL cannot be empty.")
        return

    print(f"\nFetching article from {url}...")
    try:
        article = extract_from_url(url)
        print(f"Extracted: '{article.title}' ({article.word_count} words)")
        model = prompt_model_selection(selected_model)
        run_summarization_pipeline(
            raw_text=article.text,
            model_id=model,
            source_url=article.url,
            title_hint=article.title,
        )
    except SummarizerException as exc:
        print(f"\n[EXTRACTION ERROR] {exc}\n")


def handle_summarize_file(selected_model: str) -> None:
    """Workflow for loading and summarizing a local text file."""
    path_str = input("\nEnter path to local article file (e.g. article.txt): ").strip()
    if not path_str:
        print("\n[WARNING] Path cannot be empty.")
        return

    try:
        file_path = Path(path_str)
        text = read_text_file(file_path)
        print(f"Loaded {len(text.split())} words from {file_path.name}")
        model = prompt_model_selection(selected_model)
        run_summarization_pipeline(
            raw_text=text,
            model_id=model,
            title_hint=file_path.stem.replace("_", " ").title(),
        )
    except SummarizerException as exc:
        print(f"\n[FILE ERROR] {exc}\n")


def interactive_menu(default_model: str = DEFAULT_MODEL_ID) -> None:
    """Run the interactive terminal menu loop."""
    current_model = default_model

    while True:
        print_banner()
        print(f"Active Model: {current_model}\n")
        print("Please choose an option:")
        print(" [1] Summarize pasted article text")
        print(" [2] Summarize an article from a URL")
        print(" [3] Summarize an article from a local text file")
        print(" [4] View available Gemini models")
        print(" [5] View application information")
        print(" [6] Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            handle_summarize_pasted(current_model)
            input("Press [Enter] to continue...")
        elif choice == "2":
            handle_summarize_url(current_model)
            input("Press [Enter] to continue...")
        elif choice == "3":
            handle_summarize_file(current_model)
            input("Press [Enter] to continue...")
        elif choice == "4":
            display_models_catalog()
            current_model = prompt_model_selection(current_model)
            input("Press [Enter] to continue...")
        elif choice == "5":
            display_app_info()
            input("Press [Enter] to continue...")
        elif choice == "6" or choice.lower() in ("exit", "quit", "q"):
            print("\nThank you for using AI Article Summarizer. Goodbye!\n")
            break
        else:
            print("\n[WARNING] Invalid selection. Please enter a number from 1 to 6.")
            time.sleep(1)


def main() -> None:
    """Entry point parsing CLI arguments or launching the interactive menu."""
    parser = argparse.ArgumentParser(
        description="AI Article Summarizer: Production-grade CLI tool powered by Google Gemini.",
        epilog="Examples:\n  python main.py\n  python main.py --url https://example.com/article\n  python main.py --input-file notes.txt --model gemini-3.5-flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL_ID,
        help=f"Gemini model ID to use (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--url",
        "-u",
        help="Summarize an article extracted from a web URL",
    )
    parser.add_argument(
        "--input-file",
        "-f",
        help="Summarize an article loaded from a local text file",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Directly pass article text string via command line argument",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to terminal without saving a Markdown file to outputs/",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Custom directory path to save Markdown outputs",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Display all available Gemini models and exit",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display application information and exit",
    )

    args = parser.parse_args()

    # Informational flags
    if args.list_models:
        display_models_catalog()
        sys.exit(0)

    if args.info:
        display_app_info()
        sys.exit(0)

    target_model = args.model
    custom_output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    save_output = not args.no_save

    # Non-interactive CLI executions
    try:
        if args.url:
            print_banner()
            print(f"Fetching URL: {args.url}")
            article = extract_from_url(args.url)
            run_summarization_pipeline(
                raw_text=article.text,
                model_id=target_model,
                source_url=article.url,
                title_hint=article.title,
                save_output=save_output,
                output_dir=custom_output_dir,
            )
            return

        if args.input_file:
            print_banner()
            file_path = Path(args.input_file)
            print(f"Reading file: {file_path}")
            text = read_text_file(file_path)
            run_summarization_pipeline(
                raw_text=text,
                model_id=target_model,
                title_hint=file_path.stem.replace("_", " ").title(),
                save_output=save_output,
                output_dir=custom_output_dir,
            )
            return

        if args.text:
            print_banner()
            run_summarization_pipeline(
                raw_text=args.text,
                model_id=target_model,
                save_output=save_output,
                output_dir=custom_output_dir,
            )
            return

    except SummarizerException as exc:
        print(f"\n[ERROR] Execution Failed: {exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected Error: {exc}\n")
        sys.exit(1)

    # Fall back to interactive menu when no direct execution flags provided
    try:
        interactive_menu(default_model=target_model)
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
