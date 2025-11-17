import logging
from typing import Optional

import structlog


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structured logging."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None):
    """Return a configured structlog logger."""
    return structlog.get_logger(name)
