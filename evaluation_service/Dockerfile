FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create cache directory for Hugging Face models
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Fix for oneDNN/ARM compatibility issues
# Disable oneDNN optimizations which cause issues on ARM
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV DNNL_ENABLE=0
ENV TF_ENABLE_ONEDNN_OPTS=0

RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')"

# Run the application
CMD ["python", "main.py"]
