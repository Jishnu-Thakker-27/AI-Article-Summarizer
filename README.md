# AI Article Summarizer 🧠📰

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Vite](https://img.shields.io/badge/Vite-v6+-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![React](https://img.shields.io/badge/React-v19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-v2.3.0-orange)](https://ai.google.dev/gemini-api/docs)
[![Tests](https://img.shields.io/badge/pytest-53%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, full-stack AI Article Summarizer built with **FastAPI**, **Vite + React**, Python, and Google's official **GenAI SDK** (`google-genai`). Features a modern interactive web application, REST API with real-time SSE streaming, interactive CLI, latency profiling, and map-reduce summarization for long-form literature.

---

## 🌟 Key Features

- **Modern Web Application & API**:
  - **FastAPI Backend**: Asynchronous REST endpoints, Server-Sent Events (SSE) live token streaming, file uploads, text analysis, and output history archive.
  - **Vite + React Frontend**: Sleek dark UI with live token streaming, URL extractor, file drag-and-drop, model selector, markdown and structured JSON card views, latency dashboard, and output downloads.
- **Multi-Source Input Ingestion**:
  - **Pasted Text**: Interactive multi-line terminal input with `END` sentinel terminator or web text area.
  - **Web URL Extraction**: Automated webpage parsing using `requests` and `BeautifulSoup4`, prioritizing semantic HTML (`<article>`, `<main>`), stripping clutter/ads, and respecting `robots.txt`.
  - **Local Files**: Reads `.txt` and `.md` files with automated encoding detection (`utf-8`, `latin-1`).
- **Structured Intelligence Output**:
  - Identifiable or descriptive **Article Title**
  - Crisp **TL;DR** executive summary
  - Factual, coherent **Summary** in 1–2 paragraphs
  - **Exactly 3 Key Takeaways** (non-repetitive, insightful)
  - **Important Topics & Keywords**
  - Single overarching **Central Message**
- **Real-Time Streaming Generation**:
  - Live token streaming to both web browser (via SSE) and terminal via `generate_content_stream()`.
  - Responsive user experience with progressive output display.
- **Precision Performance Profiling**:
  - Measures **Time to First Chunk (TTFT)** latency.
  - Calculates total generation duration and processing throughput (tokens/second).
  - Extracts prompt and candidate token counts from Gemini's `usage_metadata`.
- **Map-Reduce Architecture for Long Articles**:
  - Automatically detects long articles exceeding single-pass context thresholds (>3,500 words).
  - Chunks content across paragraph boundaries with context-preserving sliding overlaps.
  - Executes intermediate section summaries (**Map**) before streaming unified master synthesis (**Reduce**).
- **Prompt Injection Defense**:
  - Treats all source article text as passive, untrusted data inside explicit delimiters.
  - System instructions strictly prohibit the LLM from executing commands embedded in user-supplied text.
- **Automated Markdown Artifact Export**:
  - Persists summaries to `outputs/` with collision-proof timestamp naming (`summary_YYYYMMDD_HHMMSS.md`).
  - Embeds complete metadata (Source URL, Model, Word Count, Timing, Tokens).
- **Comprehensive Unit Testing**:
  - 53 automated unit tests covering FastAPI endpoints, validation, text normalization, chunking, web scraping, and Gemini API error mapping.

---

## 🏗️ System Architecture

```
AI-Article-Summarizer/
│
├── frontend/                 # Modern Vite + React + TypeScript web app
│   ├── src/
│   │   ├── App.tsx           # Full-featured UI: Stream, Cards, URL extractor, History
│   │   ├── index.css         # Modern typography, glassmorphism, markdown styles
│   │   └── main.tsx          # React application entrypoint
│   ├── dist/                 # Production-built bundle (served by FastAPI)
│   ├── package.json          # Node dependencies (React 19, Lucide, Marked)
│   └── vite.config.ts        # Vite config with /api reverse proxy
│
├── src/
│   ├── api/                  # FastAPI backend module
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI app, SSE streaming, file upload, static mounting
│   │   └── schemas.py        # Pydantic request & response schemas
│   ├── __init__.py           # Package marker
│   ├── main.py               # Application CLI controller & interactive menu
│   ├── config.py             # Centralized settings, model catalogs, thresholds
│   ├── gemini_service.py     # Official google-genai SDK integration & streaming
│   ├── article_processor.py  # Text cleaning, token estimation, paragraph chunker
│   ├── article_extractor.py  # Webpage scraping, semantic HTML parser, robots.txt
│   ├── prompt_templates.py   # Guardrailed prompt engineering & Pydantic schemas
│   ├── file_handler.py       # UTF-8 markdown export & collision avoidance
│   ├── validators.py         # URL, file path, text length, and key validators
│   ├── performance.py        # Latency (TTFT), duration, and throughput metrics
│   └── exceptions.py         # Descriptive, hierarchical exception classes
│
├── outputs/                  # Auto-generated summary markdown artifacts
│   └── .gitkeep
│
├── tests/                    # Pytest test suite (53 tests)
│   ├── __init__.py
│   ├── test_api.py           # FastAPI endpoint tests (health, extract, summarize, etc.)
│   ├── test_validators.py
│   ├── test_article_processor.py
│   ├── test_article_extractor.py
│   ├── test_file_handler.py
│   └── test_gemini_service.py
│
├── .env                      # Local credentials (git-ignored)
├── .env.example              # Template for environment configuration
├── requirements.txt          # Production dependencies (FastAPI, uvicorn, google-genai)
├── run_server.py             # FastAPI Uvicorn server launcher
├── main.py                   # CLI launcher script
└── README.md                 # Project documentation
```

---

## ⚡ Supported Gemini Models

Models are configured centrally in `src/config.py` using official Gemini model identifiers:

| Key | Model Name | Model ID | Intended Use | Performance Profile |
|:---:|:---|:---|:---|:---|
| **1** | **Gemini 3.5 Flash** *(Default)* | `gemini-3.5-flash` | General summarization & standard articles | Balanced Speed & Quality |
| **2** | **Gemini 3.5 Flash-Lite** | `gemini-3.5-flash-lite` | Rapid digests, short articles | Maximum Throughput |
| **3** | **Gemini 3.8 Flash** | `gemini-3.8-flash` | Complex, dense, academic literature | Highest Quality Reasoning |
| **4** | **Gemini 3.6 Flash** | `gemini-3.6-flash` | News, essays, and web articles | Balanced Multimodal |
| **5** | **Gemini 3.1 Flash-Lite** | `gemini-3.1-flash-lite` | Lightweight environments | Low-Latency Lightweight |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11 or higher**
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Clone the Repository
```bash
git clone https://github.com/Jishnu-Thakker-27/AI-Article-Summarizer.git
cd AI-Article-Summarizer
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Your API Key
Copy `.env.example` to `.env` and insert your Gemini API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

---

## 💻 Usage

### 🌐 Modern Web Interface (FastAPI + Vite)

You can launch the complete full-stack web application with the FastAPI backend and modern Vite React UI:

#### 1. Start the FastAPI Backend Server
```bash
python run_server.py
```
- **Web App / UI**: `http://127.0.0.1:8000/` (automatically serves built Vite UI)
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc API Documentation**: `http://127.0.0.1:8000/redoc`

#### 2. Start the Vite Frontend (Development Mode)
```bash
cd frontend
npm install
npm run dev
```
- Opens at: `http://localhost:5173/` (with hot module reloading and `/api` reverse proxy to FastAPI).

---

### 🔌 REST & Streaming API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/health` | System health check, API key status, default model info |
| `GET` | `/api/models` | List all available Gemini models with roles and latency tiers |
| `POST` | `/api/extract` | Extract title and main body text from any HTTP/HTTPS URL |
| `POST` | `/api/analyze` | Calculate word count, reading time, tokens, and chunk requirements |
| `POST` | `/api/upload` | Upload `.txt` or `.md` files for immediate extraction and analysis |
| `POST` | `/api/summarize` | Standard batch summarization (supports markdown & structured JSON) |
| `POST` | `/api/summarize/stream` | **Real-time SSE token streaming** directly from Gemini to browser |
| `GET` | `/api/history` | List recently generated summary files in `outputs/` |
| `GET` | `/api/history/{filename}` | Download or view a saved Markdown summary |

---

### Interactive Menu Mode
Simply run the root application launcher without arguments:
```bash
python main.py
```
This opens the terminal interface:
```
==================================================
           AI ARTICLE SUMMARIZER
         Powered by Google Gemini
==================================================

Active Model: gemini-3.5-flash

Please choose an option:
 [1] Summarize pasted article text
 [2] Summarize an article from a URL
 [3] Summarize an article from a local text file
 [4] View available Gemini models
 [5] View application information
 [6] Exit

Enter choice (1-6):
```

---

### Direct Command-Line Execution (CLI Flags)

The application supports direct, non-interactive CLI arguments using `argparse`:

#### Summarize an Online Article from a URL:
```bash
python main.py --url https://en.wikipedia.org/wiki/Artificial_intelligence
```

#### Summarize a Local Text File:
```bash
python main.py --input-file sample_article.txt
```

#### Summarize Direct Text String:
```bash
python main.py --text "Paste or pass your long text string here..."
```

#### Specify a Custom Model:
```bash
python main.py --input-file article.txt --model gemini-3.8-flash
```

#### Print to Terminal Without Saving Markdown File:
```bash
python main.py --url https://example.com/post --no-save
```

#### View Available Gemini Models & Exit:
```bash
python main.py --list-models
```

#### View Application Architecture Information:
```bash
python main.py --info
```

---

## 📊 Sample Terminal Output

```
==================================================
           AI ARTICLE SUMMARIZER
         Powered by Google Gemini
==================================================

---------------- ARTICLE DETAILS ----------------
Title            : AI Agents Revolutionize Software Development Workflows
Word Count       : 95 words
Estimated Tokens : ~156 tokens
Selected Model   : gemini-3.5-flash
-------------------------------------------------

Generating summary...

============================================================
# Title: AI Agents Revolutionize Software Development Workflows

## TL;DR
Modern AI coding agents are transforming software development by analyzing entire repository architectures and executing complex tasks, helping developers complete features up to 50% faster while requiring strong continuous integration and testing to maintain code safety.

## Summary
Artificial intelligence agents are shifting software engineering beyond simple code autocompletion toward comprehensive architecture reasoning, terminal command execution, testing, and debugging. By continuously refining implementations using real-time feedback, these intelligent tools enable developers to work across full codebase structures rather than isolated code snippets. Industry benchmarks demonstrate that developers working alongside AI agents complete features up to fifty percent faster while reporting improvements in overall code quality.

However, the adoption of AI agents requires engineering teams to implement strong safeguard mechanisms. To ensure that generated code consistently meets rigorous security and reliability standards, organizations must maintain comprehensive automated test suites and continuous integration pipelines.

## Key Takeaways
- AI coding agents have evolved beyond basic code snippet generation to reason over full repository architectures, execute terminal commands, and perform real-time debugging.
- Developers collaborating with AI agents achieve up to 50% faster feature completion alongside higher reported code quality.
- Engineering teams must maintain robust automated test suites and continuous integration pipelines to verify the security and reliability of AI-generated code.

## Important Topics & Keywords
Artificial Intelligence Agents, Software Engineering, Developer Productivity, Code Quality, Automated Testing, Continuous Integration, Repository Architecture

## Central Message
AI agents significantly boost developer speed and capabilities across complete software architectures, but engineering teams must rely on automated testing and continuous integration to guarantee security and system reliability.
============================================================

=================== PERFORMANCE ===================
Total Generation Time : 9.92 s
Time to First Chunk   : 9852 ms
Tokens (Prompt / Gen) : 495 / 330 (Total: 1703)
Throughput            : 33.3 tokens/s
===================================================

[SUCCESS] Summary report saved to: outputs\summary_20261031_200240.md
```

---

## 🧪 Running the Test Suite

Unit tests are written using `pytest` and mock network/API calls to allow offline execution:

```bash
# Run all unit tests
pytest -v

# Run with coverage report
pytest --cov=src -v
```

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.0.3
collected 44 items

tests/test_article_extractor.py::test_extract_title_og_tag PASSED        [  2%]
tests/test_article_extractor.py::test_extract_title_html_title_cleaned PASSED [  4%]
tests/test_article_extractor.py::test_extract_article_content_semantic_article PASSED [  6%]
tests/test_article_extractor.py::test_extract_from_url_success PASSED    [  9%]
tests/test_article_extractor.py::test_extract_from_url_403_forbidden PASSED [ 11%]
tests/test_article_processor.py::test_clean_and_normalize_text PASSED    [ 13%]
tests/test_article_processor.py::test_count_words PASSED                 [ 15%]
tests/test_article_processor.py::test_count_characters PASSED            [ 18%]
tests/test_article_processor.py::test_estimate_tokens PASSED             [ 20%]
tests/test_article_processor.py::test_analyze_article_valid PASSED       [ 22%]
tests/test_article_processor.py::test_analyze_article_too_short PASSED   [ 25%]
tests/test_article_processor.py::test_analyze_article_empty PASSED       [ 27%]
tests/test_article_processor.py::test_chunk_text_single_chunk PASSED     [ 29%]
tests/test_article_processor.py::test_chunk_text_multiple_chunks_with_overlap PASSED [ 31%]
tests/test_file_handler.py::test_generate_unique_filename_collision_avoidance PASSED [ 34%]
tests/test_file_handler.py::test_save_summary_markdown_structure PASSED  [ 36%]
tests/test_file_handler.py::test_read_text_file_valid PASSED             [ 38%]
tests/test_file_handler.py::test_list_saved_summaries PASSED             [ 40%]
tests/test_gemini_service.py::test_create_gemini_client_invalid_key PASSED [ 43%]
tests/test_gemini_service.py::test_map_api_exception_auth_errors PASSED  [ 45%]
tests/test_gemini_service.py::test_map_api_exception_model_not_found PASSED [ 47%]
tests/test_gemini_service.py::test_map_api_exception_rate_limits PASSED  [ 50%]
tests/test_gemini_service.py::test_map_api_exception_server_errors PASSED [ 52%]
tests/test_gemini_service.py::test_map_api_exception_timeout PASSED      [ 54%]
tests/test_gemini_service.py::test_generate_summary_mock PASSED          [ 56%]
tests/test_gemini_service.py::test_generate_summary_stream_mock PASSED   [ 59%]
tests/test_gemini_service.py::test_parse_structured_output_json PASSED   [ 61%]
tests/test_gemini_service.py::test_parse_structured_output_markdown PASSED [ 63%]
tests/test_validators.py::test_validate_url_valid PASSED                 [ 65%]
tests/test_validators.py::test_validate_url_invalid_scheme PASSED        [ 68%]
tests/test_validators.py::test_validate_url_missing_or_empty PASSED      [ 70%]
tests/test_validators.py::test_validate_url_missing_netloc PASSED        [ 72%]
tests/test_validators.py::test_validate_article_text_valid PASSED        [ 75%]
tests/test_validators.py::test_validate_article_text_empty PASSED        [ 77%]
tests/test_validators.py::test_validate_article_text_too_short PASSED    [ 79%]
tests/test_validators.py::test_validate_file_path_valid PASSED           [ 81%]
tests/test_validators.py::test_validate_file_path_not_found PASSED       [ 84%]
tests/test_validators.py::test_validate_file_path_empty_file PASSED      [ 86%]
tests/test_validators.py::test_validate_file_path_directory PASSED       [ 88%]
tests/test_validators.py::test_validate_model_id_valid PASSED            [ 90%]
tests/test_validators.py::test_validate_model_id_invalid PASSED          [ 93%]
tests/test_validators.py::test_validate_api_key_valid PASSED             [ 95%]
tests/test_validators.py::test_validate_api_key_placeholder PASSED       [ 97%]
tests/test_validators.py::test_validate_api_key_empty PASSED             [100%]

======================== 44 passed, 1 warning in 2.68s ========================
```

---

## 🛡️ Software Engineering Principles Applied

1. **Separation of Concerns**: Networking (`article_extractor.py`), API logic (`gemini_service.py`), validation (`validators.py`), text transformation (`article_processor.py`), and file storage (`file_handler.py`) are strictly decoupled.
2. **Defensive Programming**: All external inputs (user strings, remote HTML, local files, API responses) are strictly validated before processing.
3. **Security First**: API keys and secrets are never logged in plaintext or shown on terminal exceptions. Untrusted input is strictly segregated in prompt templates to prevent prompt injection.
4. **Intermediate Beginner Readability**: Code avoids convoluted metaclasses or premature enterprise abstractions, favoring type annotations, clear docstrings, and standard library conventions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
