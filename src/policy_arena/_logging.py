"""Package-level logging configuration for PolicyArena."""

from __future__ import annotations

import logging

logger = logging.getLogger("policy_arena")
logger.addHandler(logging.NullHandler())


def configure_logging(level: str | int = "INFO") -> None:
    """Configure PolicyArena logging with a console handler.

    Parameters
    ----------
    level
        Logging level name (e.g. ``"DEBUG"``, ``"INFO"``) or numeric level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if not any(
        isinstance(h, logging.StreamHandler) and h.stream.name == "<stderr>"  # type: ignore[union-attr]
        for h in logger.handlers
        if not isinstance(h, logging.NullHandler)
    ):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Update existing handler level
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.NullHandler
            ):
                h.setLevel(level)
