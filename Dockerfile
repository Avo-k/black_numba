# syntax=docker/dockerfile:1.7

# ---- builder ---------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Resolve and install deps first, separately from the source, so this layer
# is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the actual sources and install the project itself.
COPY README.md ./
COPY black_numba ./black_numba
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Note: most @njit functions in this engine don't set cache=True, so Numba's
# disk cache wouldn't survive a cross-process pre-warm anyway. The Lichess
# entry point compiles on startup (~30 s) before the event loop, which is
# fine — challenges queue up while we warm up.

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder /app /app

RUN useradd --uid 1000 --create-home bot && chown -R bot:bot /app
USER bot

CMD ["black-numba-lichess"]
