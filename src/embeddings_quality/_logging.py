"""Timed stages and console configuration for Loguru."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger


@contextmanager
def log_stage(logger: "Logger", name: str, *, level: str = "INFO") -> Iterator[None]:
    started = perf_counter()
    logger.log(level, "{}: started", name)
    try:
        yield
    except Exception:
        logger.log(level, "{}: failed after {:.2f}s", name, perf_counter() - started)
        raise
    else:
        logger.log(level, "{}: completed in {:.2f}s", name, perf_counter() - started)


@contextmanager
def cli_logging(level: str) -> Iterator[None]:
    """Configure the command-line application's sinks and clean up on exit."""
    logger.remove()
    handler = logger.add(
        sys.stderr,
        level=level,
        format="{time:HH:mm:ss} {level} {message}",
        filter="embeddings_quality",
        backtrace=False,
        diagnose=False,
    )
    logger.enable("embeddings_quality")
    try:
        yield
    finally:
        logger.remove(handler)
        logger.disable("embeddings_quality")
