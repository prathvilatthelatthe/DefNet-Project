FROM python:3.11-slim

LABEL maintainer="DeforestNet Team"
LABEL description="DeforestNet - AI-powered satellite deforestation detection system"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (CPU-only PyTorch for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy project
COPY configs/ configs/
COPY src/ src/
COPY run_api.py .
COPY run_demo.py .
COPY benchmark.py .
COPY .env.example .env

# Create necessary directories
RUN mkdir -p data/synthetic data/raw data/processed \
    models/checkpoints outputs/predictions outputs/visualizations \
    outputs/benchmark database logs

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1

# Default: start the API server with demo data
CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "5000"]
