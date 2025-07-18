"""Logger information and setup"""

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Generator

from app.config.config import settings


def setup_logger() -> None:
    """Setup function for logger called in main"""
    log_level = logging.INFO
    try:
        log_level = getattr(logging, settings.log_level.upper())
    except AttributeError as ex:
        logging.warning(
            """Failed to get and set LOG_LEVEL from config. Expected values like DEBUG, INFO, etc. 
            Exception: %s""",
            ex,
        )

    # Configure logging with a better format for CLI
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    logger: logging.Logger = logging.getLogger()
    logger.setLevel(log_level)


@contextmanager
def logger_context(
    message: str, log_level: int = logging.INFO
) -> Generator[None, None, None]:
    """Add logger context."""
    logger = logging.getLogger()
    is_enabled = logger.isEnabledFor(log_level)
    unique_id = ""
    if is_enabled:
        start_time = time.time()
        unique_id = str(uuid.uuid4())[:8]
        logger.log(log_level, msg=f"Starting ({unique_id}): {message}")
    try:
        yield
    finally:
        if is_enabled:
            elapsed_time = time.time() - start_time
            logger.log(
                log_level,
                msg=f"Finished ({unique_id}): {message} in {elapsed_time:.4f} seconds",
            )
