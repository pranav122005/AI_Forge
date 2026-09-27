# Production Dockerfile for AIForge Backend
FROM python:3.12-slim

# Application directory
WORKDIR /app

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0

# Install system dependencies
# Tesseract is required for image/OCR functionality.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the complete backend
COPY . .

# Railway/Render provides PORT at runtime.
# 8000 is the fallback for local Docker execution.
EXPOSE 8000

# Docker health check
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Start FastAPI
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
