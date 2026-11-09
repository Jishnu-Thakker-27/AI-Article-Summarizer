"""AI Article Summarizer - Application Launcher.

Launches the application via src.main.
Usage:
    python main.py
    python main.py --help
    python main.py --url https://example.com/article
    python main.py --input-file article.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    main()
