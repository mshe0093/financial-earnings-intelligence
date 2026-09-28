# ==============================================================================
# Financial Earnings Intelligence — Multi-Stage Production Dockerfile
# Optimized for minimal image size, zero credential leaks, and high performance
# ==============================================================================

# ── Stage 1: Build Dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv directly from Astral's official container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy project specification files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install production dependencies and application package into .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ── Stage 2: Minimal Runtime Container ───────────────────────────────────────
FROM python:3.11-slim AS runner

# Create non-root system user for secure container execution
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code and runtime scripts
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY pyproject.toml /app/pyproject.toml

# Configure environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DUCKDB_PATH="/app/data/earnings.duckdb"

# Create persistent data directory with non-root ownership
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to unprivileged runtime user
USER appuser

# Expose web application port (FastAPI + Gradio)
EXPOSE 8000

# Container healthcheck querying FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Launch the server script (defaults to binding 0.0.0.0:8000)
ENTRYPOINT ["python", "scripts/serve.py"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
