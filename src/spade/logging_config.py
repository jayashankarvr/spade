"""Logging configuration for SPADE."""

import logging
import sys
from typing import Optional


# Package-level logger
logger = logging.getLogger("spade")


def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """
    Configure logging for SPADE.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (default: timestamp + level + message)
        stream: Output stream (default: stderr)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if stream is None:
        stream = sys.stderr

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(format_string))

    # Configure logger
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger.

    Args:
        name: Logger name (will be prefixed with 'spade.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"spade.{name}")


# Initialize with default settings (no output unless explicitly configured)
logger.addHandler(logging.NullHandler())
