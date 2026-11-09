"""Custom exception classes for the AI Article Summarizer.

This module defines a clean, hierarchical exception system to make error handling
straightforward, predictable, and informative for intermediate Python developers.
"""


class SummarizerException(Exception):
    """Base exception for all errors raised by the AI Article Summarizer."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        """Initialize the base summarizer exception.

        Args:
            message: Explanation of what went wrong.
            hint: Optional actionable recommendation to fix the issue.
        """
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        """Format the exception message with an optional hint."""
        if self.hint:
            return f"{self.message}\nHint: {self.hint}"
        return self.message


class APIKeyError(SummarizerException):
    """Raised when the Gemini API key is missing, improperly formatted, or rejected."""

    def __init__(
        self,
        message: str = "Invalid or missing Gemini API key.",
        hint: str = (
            "Ensure GEMINI_API_KEY is set in your .env file or system environment. "
            "Get an API key from https://aistudio.google.com/app/apikey"
        ),
    ) -> None:
        super().__init__(message, hint)


class ModelNotFoundError(SummarizerException):
    """Raised when a requested Gemini model ID does not exist or is unavailable."""

    def __init__(
        self,
        model_id: str,
        hint: str = "Choose a different model from the available models list or verify your API tier.",
    ) -> None:
        message = f"Model '{model_id}' is not available or unrecognized."
        super().__init__(message, hint)


class ArticleExtractionError(SummarizerException):
    """Raised when an article cannot be extracted from a specified URL."""

    def __init__(
        self,
        message: str,
        hint: str = "Try copying and pasting the article text directly, or check if the site requires a login.",
    ) -> None:
        super().__init__(message, hint)


class ValidationError(SummarizerException):
    """Raised when user input, file path, or article content fails validation rules."""

    def __init__(
        self,
        message: str,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, hint)


class RateLimitError(SummarizerException):
    """Raised when Gemini API rate limits (RPM/TPM) are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str = "Gemini API rate limit reached.",
        hint: str = "Please wait a few moments before trying again, or check your quota at Google AI Studio.",
    ) -> None:
        super().__init__(message, hint)


class QuotaExceededError(RateLimitError):
    """Raised when Gemini API quota for the day or billing cycle is exhausted."""

    def __init__(
        self,
        message: str = "Gemini API project quota has been exhausted.",
        hint: str = "Check your quota and billing status in the Google Cloud Console / AI Studio.",
    ) -> None:
        super().__init__(message, hint)


class ContentBlockedError(SummarizerException):
    """Raised when input text or generated response violates Gemini safety policies."""

    def __init__(
        self,
        message: str = "Content generation was blocked due to safety policies.",
        hint: str = "Check if the input article contains prohibited or sensitive content.",
    ) -> None:
        super().__init__(message, hint)


class NetworkError(SummarizerException):
    """Raised when network requests fail due to timeouts or connection drops."""

    def __init__(
        self,
        message: str = "Network connection failed or timed out.",
        hint: str = "Check your internet connection and verify that Google's services are reachable.",
    ) -> None:
        super().__init__(message, hint)


class FileHandlingError(SummarizerException):
    """Raised when reading from or writing to local files fails."""

    def __init__(
        self,
        message: str,
        hint: str = "Check file path permissions and ensure the file is not locked by another process.",
    ) -> None:
        super().__init__(message, hint)
