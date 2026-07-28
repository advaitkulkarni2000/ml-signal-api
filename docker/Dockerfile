# ── DOCKERFILE ────────────────────────────────────────────────────────────────
#
# A Dockerfile is a recipe for building a container image.
# Each instruction creates a new "layer" in the image.
# Docker caches layers — if requirements.txt hasn't changed,
# it won't reinstall packages, making rebuilds fast.
#
# --- WHAT IS A CONTAINER? ---
# A container is an isolated process with its own filesystem, network,
# and process space. It's like a lightweight VM but shares the host
# kernel. The container has EXACTLY the Python version, libraries, and
# files you put in the image — nothing more, nothing less.
#
# This means: if it runs in the container on your laptop, it will run
# identically on a server, a colleague's machine, or a cloud VM.
# "It works on my machine" is solved by containers.
#
# --- MULTI-STAGE BUILDS ---
# We use two stages:
# Stage 1 (builder): install all dependencies including dev tools
# Stage 2 (runtime): copy only what's needed to run the app
# This makes the final image smaller (~500MB vs ~2GB)
# ──────────────────────────────────────────────────────────────────────────────

# ── STAGE 1: Build stage ──────────────────────────────────────────────────────
# Use an official Python image. "slim" = smaller base without dev tools.
# python:3.11-slim is ~130MB vs python:3.11 at ~350MB
FROM python:3.11-slim AS builder

# Set working directory inside the container
# All subsequent commands run from this directory
WORKDIR /app

# Install system dependencies needed to build Python packages
# - gcc: C compiler (needed by some Python packages that have C extensions)
# - curl: for downloading things if needed
# --no-install-recommends: don't install optional packages (keeps image small)
# rm -rf /var/lib/apt/lists/*: delete apt cache after install (reduces layer size)
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (before the source code)
# WHY? Docker layer caching: if only source code changes (not requirements),
# Docker reuses the cached pip install layer. This makes rebuilds much faster.
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: don't store pip cache in the image (saves ~100MB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── STAGE 2: Runtime stage ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-root user for security
# Running as root inside a container is a security risk:
# if the container is compromised, the attacker has root on the host too.
# Best practice: create a dedicated user with minimal permissions.
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed packages from builder stage
# /usr/local/lib/python3.11/site-packages = where pip installs packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY app/ ./app/
COPY training/ ./training/

# Create directories for models and MLflow data
# chown changes ownership to appuser so the app can write to these directories
RUN mkdir -p /app/models /app/mlruns && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# EXPOSE documents which port the app listens on.
# It doesn't actually open the port — that's done with docker run -p or docker-compose.
# This is documentation for people reading the Dockerfile.
EXPOSE 8000

# Environment variables
# PYTHONUNBUFFERED=1: Python output is not buffered — logs appear immediately
# PYTHONDONTWRITEBYTECODE=1: don't create .pyc files (saves space)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

# HEALTHCHECK: Docker will run this command periodically.
# If it fails, Docker marks the container as unhealthy.
# Kubernetes uses this for liveness probes.
# curl -f = fail silently on HTTP errors (exit code 1 if not 2xx)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD: the default command to run when the container starts.
# uvicorn: the ASGI server that runs FastAPI
# app.main:app = "in module app.main, find the object called app"
# --host 0.0.0.0: listen on all network interfaces (not just localhost)
#   Without this, the API is only accessible from inside the container
# --port 8000: listen on port 8000
# --workers 2: run 2 worker processes (handle 2 requests simultaneously)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
