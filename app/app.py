"""
This is the main function
"""

import logging
from typing import Any, AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse

from app.config.config import settings
from app.logger_config import logger_context, setup_logger

# Create a FastAPI application instance with title and version from settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[Any]:
    """Application lifespan cycle"""
    try:
        setup_logger()
        setup()
        yield
    finally:
        logging.info("Application shutdown complete")


app = FastAPI(
    title=settings.project_name, version=settings.project_version, lifespan=lifespan
)


@app.exception_handler(Exception)
async def fast_api_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Exception handler for unhandled exceptions"""
    logging.error("Unhandled exception: %s", str(exc))
    if isinstance(exc, HTTPException):
        raise exc
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Request failed: {str(exc)}",
    )


@app.middleware("http")
async def add_process_time_header(
    request: Request, call_next: Callable[[Request], Any]
) -> Any:
    """Add logging to all http requests"""
    async with logger_context(f"REQUEST: {request.url}"):
        return await call_next(request)


# Define a root endpoint to indicate the application's online status
@app.get("/")
async def root() -> dict[str, str]:
    """
    Root function
    """
    return {"msg": "meta_learning_mvp online"}


def setup() -> None:
    """Setup for application"""
    try:
        logging.info(
            "Starting app '%s' setup: '%s'",
            settings.project_name,
            settings.project_version,
        )
    except AttributeError as ex:  # You may still want to narrow this if possible
        logging.error("Failed to provision application on start: %s", ex)
