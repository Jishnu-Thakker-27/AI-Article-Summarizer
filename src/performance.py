"""Performance measurement and latency profiling for AI Article Summarizer.

Measures Time to First Token (TTFT) / Time to First Chunk, total generation duration,
token counts, and processing throughput without claiming unrealistic millisecond guarantees.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PerformanceMetrics:
    """Container for generation performance metrics."""

    start_time: float = field(default_factory=time.perf_counter)
    first_chunk_time: Optional[float] = None
    end_time: Optional[float] = None

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    input_word_count: int = 0
    generated_word_count: int = 0

    @property
    def total_duration(self) -> float:
        """Total elapsed time in seconds."""
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.perf_counter() - self.start_time)

    @property
    def first_chunk_latency_sec(self) -> Optional[float]:
        """Latency until the first chunk was received, in seconds."""
        if self.first_chunk_time is not None:
            return max(0.0, self.first_chunk_time - self.start_time)
        return None

    @property
    def first_chunk_latency_ms(self) -> Optional[float]:
        """Latency until the first chunk was received, in milliseconds."""
        sec = self.first_chunk_latency_sec
        return (sec * 1000.0) if sec is not None else None

    @property
    def tokens_per_second(self) -> Optional[float]:
        """Generation throughput in tokens per second."""
        if self.completion_tokens and self.total_duration > 0:
            return self.completion_tokens / self.total_duration
        return None

    @property
    def words_per_second(self) -> Optional[float]:
        """Generation throughput in words per second."""
        if self.generated_word_count and self.total_duration > 0:
            return self.generated_word_count / self.total_duration
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        return {
            "total_duration_sec": round(self.total_duration, 2),
            "first_chunk_latency_sec": (
                round(self.first_chunk_latency_sec, 3)
                if self.first_chunk_latency_sec is not None
                else None
            ),
            "first_chunk_latency_ms": (
                round(self.first_chunk_latency_ms, 1)
                if self.first_chunk_latency_ms is not None
                else None
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": (
                round(self.tokens_per_second, 1)
                if self.tokens_per_second is not None
                else None
            ),
            "input_word_count": self.input_word_count,
            "generated_word_count": self.generated_word_count,
        }


class PerformanceTracker:
    """Tracks latency, timings, and token metrics across summarization runs."""

    def __init__(self, input_word_count: int = 0) -> None:
        """Initialize the performance tracker.

        Args:
            input_word_count: Word count of the source article being summarized.
        """
        self.metrics = PerformanceMetrics(input_word_count=input_word_count)

    def record_first_chunk(self) -> None:
        """Mark the arrival of the first streamed token/chunk."""
        if self.metrics.first_chunk_time is None:
            self.metrics.first_chunk_time = time.perf_counter()

    def record_completion(self, generated_text: str = "") -> PerformanceMetrics:
        """Mark generation completion and count generated words.

        Args:
            generated_text: The final generated text string.

        Returns:
            The finalized PerformanceMetrics instance.
        """
        self.metrics.end_time = time.perf_counter()
        if generated_text:
            self.metrics.generated_word_count = len(generated_text.split())
        return self.metrics

    def record_usage_metadata(self, metadata: Any) -> None:
        """Extract and store token usage from Gemini's usage_metadata object.

        Args:
            metadata: Gemini API usage_metadata object or dictionary.
        """
        if not metadata:
            return

        # Handle object attributes (Google GenAI SDK)
        p_tokens = getattr(metadata, "prompt_token_count", None)
        c_tokens = getattr(metadata, "candidates_token_count", None)
        t_tokens = getattr(metadata, "total_token_count", None)

        # Fallback to dict keys if passed as dictionary
        if p_tokens is None and isinstance(metadata, dict):
            p_tokens = metadata.get("prompt_token_count")
            c_tokens = metadata.get("candidates_token_count")
            t_tokens = metadata.get("total_token_count")

        if p_tokens is not None:
            self.metrics.prompt_tokens = int(p_tokens)
        if c_tokens is not None:
            self.metrics.completion_tokens = int(c_tokens)
        if t_tokens is not None:
            self.metrics.total_tokens = int(t_tokens)

    def format_terminal_summary(self) -> str:
        """Format metrics into a clear, attractive terminal summary block.

        Returns:
            Multi-line string suitable for terminal display.
        """
        lines = [
            "=================== PERFORMANCE ===================",
            f"Total Generation Time : {self.metrics.total_duration:.2f} s",
        ]

        if self.metrics.first_chunk_latency_ms is not None:
            lines.append(
                f"Time to First Chunk   : {self.metrics.first_chunk_latency_ms:.0f} ms"
            )

        if self.metrics.total_tokens is not None:
            lines.append(
                f"Tokens (Prompt / Gen) : {self.metrics.prompt_tokens or 0} / "
                f"{self.metrics.completion_tokens or 0} (Total: {self.metrics.total_tokens})"
            )
            if self.metrics.tokens_per_second:
                lines.append(
                    f"Throughput            : {self.metrics.tokens_per_second:.1f} tokens/s"
                )
        elif self.metrics.generated_word_count and self.metrics.words_per_second:
            lines.append(
                f"Throughput            : {self.metrics.words_per_second:.1f} words/s"
            )

        lines.append("===================================================")
        return "\n".join(lines)
