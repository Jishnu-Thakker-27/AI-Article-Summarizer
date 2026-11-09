"""Launch the AI Article Summarizer FastAPI backend server.

Usage:
    python run_server.py
    python run_server.py --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402


def main() -> None:
    """Parse CLI arguments and start the Uvicorn ASGI server."""
    parser = argparse.ArgumentParser(description="AI Article Summarizer FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind socket host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind socket port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload on code change")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("   AI ARTICLE SUMMARIZER - FASTAPI SERVER")
    print(f"   API Docs : http://{args.host}:{args.port}/docs")
    print(f"   API Base : http://{args.host}:{args.port}/api")
    print("=" * 55 + "\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

