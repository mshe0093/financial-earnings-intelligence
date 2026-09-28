"""FastAPI backend and interactive UI dashboard."""

from earnings_intel.api.main import app, create_app
from earnings_intel.api.routes import router
from earnings_intel.api.ui import create_ui

__all__ = ["app", "create_app", "create_ui", "router"]
