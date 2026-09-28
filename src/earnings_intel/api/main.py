"""FastAPI application entrypoint with mounted Gradio interactive dashboard."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from earnings_intel.api.routes import router
from earnings_intel.api.ui import create_ui
from earnings_intel.config import get_settings
from earnings_intel.db.connection import get_connection
from earnings_intel.db.schema import initialize_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("earnings_intel.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler ensuring DuckDB schema initialization."""
    settings = get_settings()
    logger.info("Initializing DuckDB schema at %s...", settings.duckdb_path)
    try:
        with get_connection(settings) as conn:
            initialize_schema(conn)
        logger.info("DuckDB schema initialized successfully.")
    except Exception as e:
        logger.warning("Could not initialize local DuckDB file on startup: %s", e)
    yield
    logger.info("Shutting down Earnings Intelligence API.")


def create_app() -> FastAPI:
    """Create and configure the unified FastAPI and Gradio application."""
    app = FastAPI(
        title="Financial Earnings Intelligence API",
        description=(
            "LLM-powered SEC filing analysis, 12-metric structured extraction, "
            "and quantitative PEAD backtesting engine."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include REST API routes
    app.include_router(router)

    # Mount Gradio UI at root path
    ui = create_ui()
    app = gr.mount_gradio_app(app, ui, path="/")

    return app


app = create_app()
